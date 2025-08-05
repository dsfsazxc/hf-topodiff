"""
Train a time-dependent reward model to predict BC violations.
Reads BC feedback from .npy files with topology and constraint data.
"""

import argparse
import os
import numpy as np

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score

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


class BCRewardDataset(Dataset):
    """Dataset for BC reward model loading topology and constraint data"""
    
    def __init__(self, data_dir, image_size=64, enable_augmentation=False):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.enable_augmentation = enable_augmentation
        
        # Check directory existence
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        # Load BC feedback file
        feedback_path = os.path.join(data_dir, "feedback_bc.npy")
        combined_path = os.path.join(data_dir, "feedback_combined.npy")
        
        if os.path.exists(feedback_path):
            self.feedback = np.load(feedback_path)
            logger.log(f"BC feedback file loaded: {feedback_path}, size: {self.feedback.shape}")
        elif os.path.exists(combined_path):
            # Convert from combined file: combined value 1 or 3 means BC=1, else 0
            combined = np.load(combined_path)
            self.feedback = np.array([(x == 1 or x == 3) for x in combined], dtype=np.int8)
            logger.log(f"Converted from combined feedback to BC, size: {self.feedback.shape}")
        else:
            raise ValueError(f"BC feedback file not found: {feedback_path} or {combined_path}")
        
        # Create image file list
        self.image_files = []
        for i in range(len(self.feedback)):
            img_path = os.path.join(data_dir, f"gt_topo_{i}.png")
            pf_path = os.path.join(data_dir, f"cons_pf_array_{i}.npy")
            load_path = os.path.join(data_dir, f"cons_load_array_{i}.npy")
            bc_path = os.path.join(data_dir, f"cons_bc_array_{i}.npy")
            
            # Check if all required files exist
            if all(os.path.exists(p) for p in [img_path, pf_path, load_path, bc_path]):
                self.image_files.append({
                    'index': i,
                    'img_path': img_path,
                    'pf_path': pf_path,
                    'load_path': load_path,
                    'bc_path': bc_path,
                    'label': self.feedback[i]
                })
        
        # Dataset statistics
        pos_count = sum(1 for item in self.image_files if item['label'] == 1)
        neg_count = len(self.image_files) - pos_count
        
        logger.log(f"Total {len(self.image_files)} samples loaded")
        logger.log(f"Positive (violation) samples: {pos_count}")
        logger.log(f"Negative (normal) samples: {neg_count}")
        if len(self.image_files) > 0:
            logger.log(f"Positive ratio: {pos_count/len(self.image_files):.2%}")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        item = self.image_files[idx]
        
        # Load topology image
        with open(item['img_path'], 'rb') as f:
            img = Image.open(f)
            img.load()
        img = img.convert("RGB")
        
        # Image preprocessing
        arr = self._center_crop_arr(img, self.image_size)
        arr = np.mean(arr, axis=2)  # RGB to grayscale
        arr = arr.astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
        arr = arr.reshape(self.image_size, self.image_size, 1)
        
        # Load constraint data
        constraints_pf = np.load(item['pf_path'])
        loads = np.load(item['load_path'])
        bcs = np.load(item['bc_path'])
        
        # Combine constraint data
        constraints = np.concatenate([constraints_pf, loads, bcs], axis=2)
        
        # Normalize each channel
        for i in range(constraints.shape[2]):
            channel = constraints[:, :, i]
            eps = 1e-6
            constraints[:, :, i] = (channel - channel.mean()) / (channel.std() + eps)

        # Apply data augmentation
        if self.enable_augmentation:
            arr, constraints = self._apply_augmentation(arr, constraints)
        
        # Change channel order (H, W, C) -> (C, H, W)
        topology = np.transpose(arr, [2, 0, 1])
        constraint_tensor = np.transpose(constraints, [2, 0, 1]).astype(np.float32)
        
        # Convert to tensors
        topology_tensor = th.from_numpy(topology.copy()).float()
        constraint_tensor = th.from_numpy(constraint_tensor.copy()).float()
        label = th.tensor(item['label'], dtype=th.float32)
        
        return topology_tensor, constraint_tensor, {"y": label}
    
    def _center_crop_arr(self, pil_image, image_size):
        """Center crop and resize image"""
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    def _apply_augmentation(self, image_arr, constraints_arr):
        """Apply data augmentation to both image and constraints"""
        import random
        
        augmentation_types = [
            "original", "rot90", "rot180", "rot270",
            "flip_lr", "flip_ud", "flip_diag1", "flip_diag2"
        ]
        
        aug_type = random.choice(augmentation_types)
        
        if aug_type == "original":
            return image_arr.copy(), constraints_arr.copy()
        elif aug_type == "rot90":
            return (np.rot90(image_arr, k=1, axes=(0, 1)).copy(), 
                    np.rot90(constraints_arr, k=1, axes=(0, 1)).copy())
        elif aug_type == "rot180":
            return (np.rot90(image_arr, k=2, axes=(0, 1)).copy(), 
                    np.rot90(constraints_arr, k=2, axes=(0, 1)).copy())
        elif aug_type == "rot270":
            return (np.rot90(image_arr, k=3, axes=(0, 1)).copy(), 
                    np.rot90(constraints_arr, k=3, axes=(0, 1)).copy())
        elif aug_type == "flip_lr":
            return (np.fliplr(image_arr).copy(), 
                    np.fliplr(constraints_arr).copy())
        elif aug_type == "flip_ud":
            return (np.flipud(image_arr).copy(), 
                    np.flipud(constraints_arr).copy())
        elif aug_type == "flip_diag1":
            return (np.transpose(image_arr, (1, 0, 2)).copy(),
                    np.transpose(constraints_arr, (1, 0, 2)).copy())
        elif aug_type == "flip_diag2":
            # transpose + flip both axes
            temp_img = np.transpose(image_arr, (1, 0, 2))
            temp_constraints = np.transpose(constraints_arr, (1, 0, 2))
            return (temp_img[::-1, ::-1].copy(),
                    temp_constraints[::-1, ::-1].copy())


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.log_dir)
    
    logger.log("Arguments:")
    for k, v in sorted(vars(args).items()):
        logger.log(f"  {k}: {v}")

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
    
    # Create dataset
    dataset = BCRewardDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        enable_augmentation=args.enable_augmentation
    )
    
    # Split train/validation datasets
    if args.val_split > 0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        if train_size > 0 and val_size > 0:
            # Fixed random split
            generator = th.Generator().manual_seed(args.random_seed)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            
            logger.log(f"Dataset split: {train_size} training, {val_size} validation samples")
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False
            )
        else:
            # Insufficient data for validation split
            train_loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
            )
            val_loader = None
            logger.log("Cannot create validation set (insufficient data)")
    else:
        # No validation split
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
        )
        val_loader = None
        logger.log("No validation set (val_split=0)")
    
    # Create infinite training iterator
    def infinite_train():
        while True:
            for batch in train_loader:
                yield batch
    
    train_data = infinite_train()

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

    logger.log("Training BC reward model...")

    # Training progress tracking
    best_f1 = 0.0

    # Loss function
    pos_weight = th.tensor([args.pos_weight], device=dist_util.dev())
    loss_fn = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward_backward_log(data_loader, prefix="train"):
        """Forward-backward pass for training"""
        topology, constraints, extra = next(data_loader)
        labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)
        
        topology = topology.to(dist_util.dev())
        constraints = constraints.to(dist_util.dev())
        
        # Apply noise to topology only
        if args.noised:
            t, _ = schedule_sampler.sample(topology.shape[0], dist_util.dev())
            topology = diffusion.q_sample(topology, t)
        else:
            t = th.zeros(topology.shape[0], dtype=th.long, device=dist_util.dev())
        
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for i, (sub_topo, sub_cons, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, topology, constraints, labels, t)
        ):
            # Combine inputs (topology + constraints)
            full_batch = th.cat((sub_topo, sub_cons), dim=1)
            logits = model(full_batch, timesteps=sub_t)
            loss = loss_fn(th.flatten(logits), sub_labels)
            
            # Generate predictions for metrics
            probs = th.sigmoid(th.flatten(logits)).detach()
            preds = (probs > args.pred_threshold).float()
            
            # Calculate accuracy
            acc = (preds == sub_labels).float().mean().item()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sub_labels.cpu().numpy())
            
            # Weighted averaging by batch size
            total_loss += loss.item() * len(sub_topo)
            total_acc += acc * len(sub_topo)
            total_samples += len(sub_topo)

            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_topo) / len(topology))
        
        # Calculate averages
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
        if val_loader is None:
            return 0.0, 0.0, 0.0
            
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        sample_count = 0
        
        with th.no_grad():
            for topology, constraints, extra in val_loader:
                labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)
                topology = topology.to(dist_util.dev())
                constraints = constraints.to(dist_util.dev())
                
                # No noise for validation
                t = th.zeros(topology.shape[0], dtype=th.long, device=dist_util.dev())
                
                # Combine inputs
                full_batch = th.cat((topology, constraints), dim=1)
                logits = model(full_batch, timesteps=t)
                
                loss = loss_fn(th.flatten(logits), labels)
                probs = th.sigmoid(th.flatten(logits)).detach()
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                sample_count += batch_size
                
                # Store predictions and labels
                preds = (probs > args.pred_threshold).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / sample_count if sample_count > 0 else 0.0
        
        if len(all_labels) > 0 and len(all_preds) > 0:
            try:
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds)
            except:
                acc = 0.0
                f1 = 0.0
        else:
            acc = 0.0
            f1 = 0.0
        
        model.train()
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
        loss, acc, f1 = forward_backward_log(train_data)
        mp_trainer.optimize(opt)

        # Periodic logging and validation
        if step % args.log_interval == 0:
            # Evaluate validation performance
            val_loss, val_acc, val_f1 = evaluate_model(model, val_loader)
            
            # Log metrics
            logger.logkv("train_loss", loss)
            logger.logkv("train_accuracy", acc)
            logger.logkv("train_f1", f1)
            logger.logkv("val_loss", val_loss)
            logger.logkv("val_accuracy", val_acc)
            logger.logkv("val_f1", val_f1)
            logger.dumpkvs()
            
            # Console output
            if dist.get_rank() == 0:
                print(f"Step {step+resume_step}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train F1: {f1:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
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
            logger.log("Saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    # Save final model
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
        data_dir="",
        log_dir="",
        noised=True,
        iterations=40000, 
        lr=5e-5, 
        weight_decay=0.1, 
        anneal_lr=False,
        batch_size=8,  
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=1000,
        pos_weight=1.0,
        pred_threshold=0.5,
        val_split=0.2,
        random_seed=42,
        image_size=64,
        in_channels=8,  # 1 topology + 7 constraint channels
        output_dim=1,
        enable_augmentation=True,
        
        # Classifier parameters
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=4,
        classifier_attention_resolutions="32,16,8",
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="spatial",
    )
    
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    return parser


if __name__ == "__main__":
    main()