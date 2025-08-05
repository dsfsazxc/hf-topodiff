"""
Train a time-dependent reward model from human feedback data.
Enhanced with validation tracking, visualization, and best model saving.
Reads FM feedback from .npy files.
"""

import argparse
import os
import pickle
import random
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

from HFtopodiff import dist_util, logger
from HFtopodiff.fp16_util import MixedPrecisionTrainer
from HFtopodiff.resample import create_named_schedule_sampler
from HFtopodiff.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from HFtopodiff.train_util import parse_resume_step_from_filename, log_loss_dict

class FMRewardFileDataset(Dataset):
    def __init__(
        self,
        feedback_dict,
        rgb=False,
    ):
        super().__init__()
        
        image_paths = []
        labels = []
        
        for path in feedback_dict.keys():
            if feedback_dict[path] is not None:
                image_paths.append(path)
                labels.append(feedback_dict[path])
        
        self.local_images = image_paths
        self.local_classes = th.tensor(labels, dtype=int)
        
        self.rgb = rgb

    def augment_dataset(self, benign_transform, malign_transform, augment_data_dir, num_augment,):
        
        if augment_data_dir is None:
            any_img_path = self.local_images[0]
            augment_data_dir = os.path.join(any_img_path[:any_img_path.find("/", -1)], "temp")
        if not os.path.isdir(augment_data_dir) and dist.get_rank() == 0:
            os.makedirs(augment_data_dir)
        
        dist.barrier()

        new_paths = []
        for idx, path in enumerate(self.local_images):
            with bf.BlobFile(path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            temp_path = os.path.join(augment_data_dir, f"{idx:04d}.png")
            pil_image.save(temp_path)
            new_paths.append(temp_path)
        
        for aug_loop in range(num_augment):
            for idx, path in enumerate(self.local_images):
                with bf.BlobFile(path, "rb") as f:
                    pil_image = Image.open(f)
                    pil_image.load()
                new_idx = (1 + aug_loop) * len(self.local_images) + idx
                temp_path = os.path.join(augment_data_dir, f"{new_idx:04d}.png")
                if self.local_classes[idx] == 0:
                    malign_transform(pil_image).save(temp_path)
                elif self.local_classes[idx] == 1:
                    benign_transform(pil_image).save(temp_path)
                new_paths.append(temp_path)
        
        self.local_images = new_paths
        self.local_classes = self.local_classes.repeat(num_augment + 1)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            if not self.rgb:
                pil_image = pil_image.convert("L")
            pil_image.load()

        arr = np.array(pil_image)
        arr = arr.astype(np.float32) / 127.5 - 1
        if not self.rgb:
            arr = np.expand_dims(arr, axis=2)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

def load_fm_feedback_from_npy(data_dir):
    """
    Load FM feedback data from .npy files and convert to dictionary format
    compatible with FMRewardFileDataset
    """
    fm_feedback_path = os.path.join(data_dir, "feedback_fm.npy")
    if not os.path.exists(fm_feedback_path):
        raise FileNotFoundError(f"FM feedback file not found: {fm_feedback_path}")
    
    fm_feedback = np.load(fm_feedback_path)
    logger.log(f"Loaded FM feedback: {len(fm_feedback)} samples")
    
    # Create dictionary mapping image paths to feedback
    feedback_dict = {}
    
    for i in range(len(fm_feedback)):
        img_path = os.path.join(data_dir, f"gt_topo_{i}.png")
        if os.path.exists(img_path):
            # Log sample image info
            if i == 0:
                from PIL import Image
                test_img = Image.open(img_path)
                logger.log(f"Sample image: mode={test_img.mode}, size={test_img.size}")
                img_array = np.array(test_img)
                logger.log(f"Image array shape: {img_array.shape}")
                test_img.close()
            
            # FM feedback: 1=negative(floating material), 0=normal
            # FMRewardFileDataset expects 0=negative, 1=normal, so invert
            feedback_dict[img_path] = 1 - int(fm_feedback[i])
        else:
            logger.log(f"Warning: Image file not found: {img_path}")
    
    logger.log(f"Created feedback dictionary with {len(feedback_dict)} images")
    logger.log(f"FM positive (floating material): {sum(fm_feedback)}")
    logger.log(f"Normal samples: {sum(1-fm_feedback)}")
    
    return feedback_dict


def main():
    args = create_argparser().parse_args()
    args.image_channels = args.in_channels
    
    dist_util.setup_dist()
    log_dir = args.log_dir
    logger.configure(log_dir)
    logger.log(vars(args))

    logger.log("Creating model and diffusion...")

    # Create model with proper arguments
    base_keys = list(classifier_and_diffusion_defaults().keys())
    call_kwargs = args_to_dict(args, base_keys)
    call_kwargs['in_channels'] = args.in_channels
    call_kwargs['output_dim'] = args.output_dim
    model, diffusion = create_classifier_and_diffusion(**call_kwargs)

    model.to(dist_util.dev())
    
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        logger.log(f"Loading model from checkpoint: {args.resume_checkpoint} at step {resume_step}")
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )
        logger.log("Model loaded successfully!")

    # Sync parameters for correct EMAs and fp16
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("Creating data loader...")

    def load_from_feedback(feedback_dict, batch_size, val_split=None):
        """Create train/validation data loaders from feedback dictionary"""
        if val_split is None:
            val_split = args.val_split
            
        # Create full dataset
        dataset = FMRewardFileDataset(
            feedback_dict=feedback_dict,
            rgb=args.rgb,
        )
        
        # Apply geometric data augmentation (8 transformations)
        if args.enable_augmentation:
            from PIL import Image
            
            def apply_augmentation(image_arr, aug_type):
                """Apply one of 8 geometric transformations"""
                if aug_type == "original":
                    return image_arr.copy()
                elif aug_type == "rot90":
                    return np.rot90(image_arr, k=1, axes=(0, 1)).copy()
                elif aug_type == "rot180":
                    return np.rot90(image_arr, k=2, axes=(0, 1)).copy()
                elif aug_type == "rot270":
                    return np.rot90(image_arr, k=3, axes=(0, 1)).copy()
                elif aug_type == "flip_lr":
                    return np.fliplr(image_arr).copy()
                elif aug_type == "flip_ud":
                    return np.flipud(image_arr).copy()
                elif aug_type == "flip_diag1":
                    return np.transpose(image_arr, (1, 0, 2)).copy()
                elif aug_type == "flip_diag2":
                    # transpose + flip both axes
                    temp_img = np.transpose(image_arr, (1, 0, 2))
                    return temp_img[::-1, ::-1].copy()
                else:
                    return image_arr.copy()
            
            def numpy_augment_transform(pil_image):
                """Apply random geometric augmentation to PIL image"""
                # PIL to numpy
                img_arr = np.array(pil_image)
                if len(img_arr.shape) == 2:
                    img_arr = img_arr[:, :, np.newaxis]  # grayscale handling
                
                # Random selection from 8 transformations
                aug_types = ["original", "rot90", "rot180", "rot270", 
                           "flip_lr", "flip_ud", "flip_diag1", "flip_diag2"]
                aug_type = random.choice(aug_types)
                
                # Apply transformation
                augmented_arr = apply_augmentation(img_arr, aug_type)
                
                # numpy to PIL
                if augmented_arr.shape[2] == 1:
                    augmented_arr = augmented_arr.squeeze(2)  # grayscale
                return Image.fromarray(augmented_arr.astype(np.uint8))
            
            augment_transform = T.Lambda(numpy_augment_transform)
            
            dataset.augment_dataset(
                benign_transform=augment_transform,
                malign_transform=augment_transform,
                augment_data_dir=args.augment_data_dir,
                num_augment=8,  # Use all 8 transformations
            )

        # Split train/validation datasets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        # Fixed random split with configurable seed
        generator = th.Generator().manual_seed(args.random_seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
        
        logger.log(f"Dataset split: {train_size} training, {val_size} validation samples")
        
        # Training data loader (infinite loop)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=1, drop_last=False
        )
        
        # Validation data loader
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=1, drop_last=False
        )
        
        # Infinite training data generator
        def infinite_train():
            while True:
                for batch in train_loader:
                    yield batch
                    
        return infinite_train(), val_loader

    # Load FM feedback from .npy files
    if args.data_dir:
        feedback_dict = load_fm_feedback_from_npy(args.data_dir)
    else:
        # Fallback: load from pickle file (for compatibility)
        with open(args.feedback_path, "rb") as f:
            feedback_dict = pickle.load(f)
    
    # Create data loaders
    train_data, val_loader = load_from_feedback(feedback_dict, args.batch_size)

    logger.log("Creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"Loading optimizer state from: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("Training reward model...")

    # Loss function with pos_weight
    loss_fn = th.nn.BCEWithLogitsLoss(
        pos_weight=args.pos_weight * th.ones([1], device=dist_util.dev())
    )
    
    # Training progress tracking
    best_val_loss = float('inf')
    best_f1 = 0.0
    
    def forward_backward_log(data_loader, prefix="train"):
        """Forward-backward pass for training"""
        batch, extra = next(data_loader)
        labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)

        batch = batch.to(dist_util.dev())
        # Add noise if specified
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            loss = loss_fn(th.flatten(logits), sub_labels)
            
            # Generate predictions for accuracy and F1 calculation
            probs = th.sigmoid(th.flatten(logits)).detach()
            preds = (probs > args.pred_threshold).float()
            
            # Calculate accuracy
            acc = (preds == sub_labels).float().mean().item()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sub_labels.cpu().numpy())
            
            # Apply batch size weighted averaging
            total_loss += loss.item() * len(sub_batch)
            total_acc += acc * len(sub_batch)
            total_samples += len(sub_batch)

            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        # Calculate F1 score
        try:
            f1 = f1_score(all_labels, all_preds)
        except:
            f1 = 0.0
            
        return avg_loss, avg_acc, f1
    
    def evaluate_model(model, val_loader):
        """Evaluate model on validation set"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with th.no_grad():
            for batch, extra in val_loader:
                labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)
                batch = batch.to(dist_util.dev())
                
                # No noise for validation
                t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
                
                logits = model(batch, timesteps=t)
                loss = loss_fn(th.flatten(logits), labels)
                
                # Generate predictions
                probs = th.sigmoid(th.flatten(logits)).detach()
                preds = (probs > args.pred_threshold).float()
                
                total_loss += loss.item() * labels.size(0)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader.dataset)
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        try:
            f1 = f1_score(all_labels, all_preds)
        except:
            f1 = 0.0
        
        model.train()  # Back to training mode
        return avg_loss, acc, f1

    # Training loop
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        # Training step
        train_loss, train_acc, train_f1 = forward_backward_log(train_data)
        mp_trainer.optimize(opt)

        # Periodic logging and validation
        if not step % args.log_interval:
            # Evaluate validation performance
            val_loss, val_acc, val_f1 = evaluate_model(model, val_loader)
            
            # Log metrics
            logger.logkv("train_loss", train_loss)
            logger.logkv("train_accuracy", train_acc)
            logger.logkv("train_f1", train_f1)
            logger.logkv("val_loss", val_loss)
            logger.logkv("val_accuracy", val_acc)
            logger.logkv("val_f1", val_f1)
            logger.dumpkvs()
            
            # Console output
            if dist.get_rank() == 0:
                print(f"Step {step+resume_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best models (two criteria)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if dist.get_rank() == 0:
                    save_model(mp_trainer, opt, step + resume_step, suffix="best_loss")
                    logger.log(f"New best validation loss: {best_val_loss:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                if dist.get_rank() == 0:
                    save_model(mp_trainer, opt, step + resume_step, suffix="best_f1")
                    logger.log(f"New best F1 score: {best_f1:.4f}")

        # Periodic model saving
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("Saving model checkpoint...")
            save_model(mp_trainer, opt, step + resume_step)

    # Save final model after training
    if dist.get_rank() == 0:
        logger.log("Saving final model...")
        save_model(mp_trainer, opt, step + resume_step)
    
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    """Set annealed learning rate"""
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, suffix=None):
    """Save model with optional suffix"""
    if dist.get_rank() == 0:
        # Save model checkpoint
        filename = f"model{step:06d}.pt" if suffix is None else f"model_{suffix}.pt"
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), filename),
        )
        
        # Save optimizer state
        opt_filename = f"opt{step:06d}.pt" if suffix is None else f"opt_{suffix}.pt"
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), opt_filename))


def split_microbatches(microbatch, *args):
    """Split batch into microbatches"""
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",  # Directory containing .npy feedback files
        log_dir="",
        noised=True,
        iterations=20000,
        lr=1e-4,
        weight_decay=0.05,
        anneal_lr=False,
        batch_size=32,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=1000,
        rgb=False,
        image_size=64,
        in_channels=1,  # 1 topology
        output_dim=1,
        feedback_path="", 
        pos_weight=1.0, 
        pred_threshold=0.5,  
        val_split=0.2,  
        random_seed=42,  
        enable_augmentation=False,  
        augment_data_dir=None,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()