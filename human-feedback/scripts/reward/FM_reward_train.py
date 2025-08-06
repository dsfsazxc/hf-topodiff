"""
Train a time-dependent reward model to predict FM violations.
Support for distributed training.
"""
import torch.nn as nn
import argparse
import os
import numpy as np
from mpi4py import MPI

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class FMRewardFileDataset(Dataset):
    """Dataset for FM reward model that loads individual files with distributed training support"""
    
    def __init__(self, data_dir, image_size=64, rgb=False, 
                 shard=0, num_shards=1, enable_augmentation=False, 
                 save_augmentation=False, augment_data_dir=None):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.rgb = rgb
        self.shard = shard
        self.num_shards = num_shards
        self.enable_augmentation = enable_augmentation
        self.save_augmentation = save_augmentation
        self.augment_data_dir = augment_data_dir
        
        if self.save_augmentation and self.augment_data_dir and self.shard == 0:
            self._initialize_augment_directory()
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
            
        # Load FM feedback file
        feedback_path = os.path.join(data_dir, "feedback_fm.npy")
        combined_path = os.path.join(data_dir, "feedback_combined.npy")
        
        if os.path.exists(feedback_path):
            self.feedback = np.load(feedback_path)
            logger.log(f"FM feedback file loaded: {feedback_path}, size: {self.feedback.shape}")
        elif os.path.exists(combined_path):
            # FM violation: combined value 1 or 3 means FM=1, else 0
            combined = np.load(combined_path)
            self.feedback = np.array([(x == 1 or x == 3) for x in combined], dtype=np.int8)
            logger.log(f"Converted from combined feedback to FM, size: {self.feedback.shape}")
        else:
            raise ValueError(f"FM feedback file not found: {feedback_path} or {combined_path}")
        
        # Create image file list (FM only needs topology images)
        all_image_files = []
        for i in range(len(self.feedback)):
            img_path = os.path.join(data_dir, f"gt_topo_{i}.png")
            
            if os.path.exists(img_path):
                all_image_files.append({
                    'index': i,
                    'img_path': img_path,
                    'label': self.feedback[i]
                })
        
        # Shard data for distributed training
        self.image_files = all_image_files[self.shard::self.num_shards]
        
        # Dataset statistics
        pos_count = sum(1 for item in self.image_files if item['label'] == 1)
        neg_count = len(self.image_files) - pos_count
        
        logger.log(f"Rank {self.shard}/{self.num_shards} - Total {len(self.image_files)} samples loaded")
        logger.log(f"Positive samples: {pos_count}, Negative samples: {neg_count}")

    def _initialize_augment_directory(self):
        if not (self.save_augmentation and self.augment_data_dir and self.shard == 0):
            return
        
        try:
            if os.path.exists(self.augment_data_dir):
                existing_files = len([f for f in os.listdir(self.augment_data_dir) 
                                    if f.startswith('aug_')])
                
                if existing_files > 0:
                    import shutil
                    shutil.rmtree(self.augment_data_dir)
            
            os.makedirs(self.augment_data_dir, exist_ok=True)
            
            init_file = os.path.join(self.augment_data_dir, ".initialized")
            with open(init_file, 'w') as f:
                f.write("Augmentation directory initialized\n")
                
        except Exception as e:
            logger.log(f"Augmentation directory initialization error: {e}")
            raise

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        item = self.image_files[idx]
        original_idx = item['index']
        
        # Load topology image
        with open(item['img_path'], 'rb') as f:
            img = Image.open(f)
            img.load()
        
        # Convert to grayscale if not RGB
        if not self.rgb:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        
        # Image preprocessing
        arr = self._center_crop_arr(img, self.image_size)
        arr = arr.astype(np.float32) / 127.5 - 1  # Normalize to [-1, 1]
        
        # Handle channel dimension
        if not self.rgb:
            arr = np.expand_dims(arr, axis=2)  # Add channel dimension for grayscale
        
        # Apply data augmentation
        aug_type = "original"
        if self.enable_augmentation:
            arr, aug_type = self._apply_augmentation(arr)
            
            if self.save_augmentation and self.augment_data_dir and self.shard == 0:
                self._save_augmented_data(arr, aug_type, original_idx)
        
        # Change channel order (H, W, C) -> (C, H, W)
        topology = np.transpose(arr, [2, 0, 1])
        
        # Convert to tensors
        topology_tensor = th.from_numpy(topology.copy()).float()
        label = th.tensor(item['label'], dtype=th.float32)
        
        return topology_tensor, {"y": label}
    
    def _center_crop_arr(self, pil_image, image_size):
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
    
    def _apply_augmentation(self, image_arr):
        import random
        
        augmentation_types = [
            "original", "rot90", "rot180", "rot270",
            "flip_lr", "flip_ud",
            "flip_diag1", "flip_diag2"
        ]
        
        aug_type = random.choice(augmentation_types)
        
        if aug_type == "original":
            return image_arr.copy(), aug_type
        elif aug_type == "rot90":
            return np.rot90(image_arr, k=1, axes=(0, 1)).copy(), aug_type
        elif aug_type == "rot180":
            return np.rot90(image_arr, k=2, axes=(0, 1)).copy(), aug_type
        elif aug_type == "rot270":
            return np.rot90(image_arr, k=3, axes=(0, 1)).copy(), aug_type
        elif aug_type == "flip_lr":
            return np.fliplr(image_arr).copy(), aug_type
        elif aug_type == "flip_ud":
            return np.flipud(image_arr).copy(), aug_type
        elif aug_type == "flip_diag1":
            return np.transpose(image_arr, (1, 0, 2)).copy(), aug_type
        elif aug_type == "flip_diag2":
            temp_img = np.transpose(image_arr, (1, 0, 2))
            return temp_img[::-1, ::-1].copy(), aug_type

    def _save_augmented_data(self, aug_image, aug_type, original_idx):
        if not self.save_augmentation:
            return
            
        try:
            base_name = f"aug_{aug_type}_{original_idx}"
            
            # Save image
            if aug_image.shape[2] == 1:  # Grayscale
                img_denorm = ((aug_image.squeeze() + 1) * 127.5).astype(np.uint8)
                img_pil = Image.fromarray(img_denorm, mode='L')
            else:  # RGB
                img_denorm = ((aug_image + 1) * 127.5).astype(np.uint8)
                img_pil = Image.fromarray(img_denorm)
            
            img_path = os.path.join(self.augment_data_dir, f"{base_name}.png")
            img_pil.save(img_path)
            
        except Exception as e:
            logger.log(f"Augmented data save error: {e}")


def setup_gpu_for_distributed():
    """Setup GPU for distributed training"""
    import os
    import torch
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        available_gpus = visible_devices.split(',')
        if len(available_gpus) >= world_size:
            gpu_id = available_gpus[rank]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        
    if torch.cuda.is_available():
        torch.cuda.set_device(0)


def main():
    args = create_argparser().parse_args()

    setup_gpu_for_distributed()
    
    dist_util.setup_dist()
    logger.configure(args.log_dir)
    
    logger.log("creating model and diffusion...")
    
    required_keys = list(classifier_and_diffusion_defaults().keys()) + [
        'in_channels', 
        'classifier_depth', 
        'output_dim'
    ]
    args_dict = args_to_dict(args, required_keys)
    args_dict['in_channels'] = args.in_channels
    args_dict['classifier_depth'] = args.classifier_depth
    args_dict['output_dim'] = args.output_dim
    
    args_dict['output_dim'] = 1
    
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    try:
        model, diffusion = create_classifier_and_diffusion(**args_dict)
        model.apply(init_weights)
        logger.log("Model creation successful")
    except Exception as e:
        logger.log(f"Error creating model: {e}")
        raise
    
    model.to(dist_util.dev())
    
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {args.resume_checkpoint} at step {resume_step}")
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )

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

    logger.log("creating data loader...")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    dataset = FMRewardFileDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        rgb=args.rgb,
        shard=rank,
        num_shards=world_size,
        enable_augmentation=args.enable_augmentation,
        save_augmentation=args.save_augmentation,
        augment_data_dir=args.augment_data_dir
    )
    
    if args.val_split > 0:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        if train_size > 0 and val_size > 0:
            # Fixed random split
            generator = th.Generator().manual_seed(args.random_seed)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
            )
            val_loader = None
    else:
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=False
        )
        val_loader = None
    
    def infinite_train():
        while True:
            if train_loader is not None:
                for batch in train_loader:
                    yield batch
            else:
                yield None, {"y": None}
    
    train_data = infinite_train()

    logger.log("creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training FM reward model...")

    best_f1 = 0.0

    pos_weight = th.tensor([args.pos_weight], device=dist_util.dev())
    loss_fn = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward_backward_log(data_loader, prefix="train"):
        batch_data = next(data_loader)
        
        if batch_data[0] is None or batch_data[1]["y"] is None:
            return 0.0, 0.0, 0.0
            
        batch, extra = batch_data
        labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)
        
        batch = batch.to(dist_util.dev())
        
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        
        losses = {}
        acc_value = 0.0
        f1_value = 0.0
        accumulated_loss = 0.0
        total_batches = 0
        
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            
            loss = loss_fn(th.flatten(logits), sub_labels) + 1e-8
            current_loss_value = loss.item()
            
            probs = th.sigmoid(th.flatten(logits)).detach()
            preds = (probs > args.pred_threshold).float()
            acc_value = (preds == sub_labels).float().mean().item()
            
            y_true = sub_labels.cpu().numpy()
            y_pred = preds.cpu().numpy()
            try:
                f1_value = f1_score(y_true, y_pred, zero_division=0)
            except:
                f1_value = 0.0
            
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_accuracy"] = th.tensor(acc_value, device=dist_util.dev())
            losses[f"{prefix}_f1"] = th.tensor(f1_value, device=dist_util.dev())

            log_loss_dict(diffusion, sub_t, losses)
            
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
            
            accumulated_loss += current_loss_value
            total_batches += 1
        
        final_loss = accumulated_loss / total_batches if total_batches > 0 else 0.0
        
        metrics = th.tensor([final_loss, acc_value, f1_value], device=dist_util.dev())
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= dist.get_world_size()
        
        return metrics[0].item(), metrics[1].item(), metrics[2].item()

    def evaluate_model(model, val_loader):
        model.eval()
        
        if val_loader is None:
            return 0.0, 0.0, 0.0
            
        total_loss = 0.0
        all_preds = []
        all_labels = []
        sample_count = 0
        
        with th.no_grad():
            for batch, extra in val_loader:
                if extra["y"] is None or len(extra["y"]) == 0:
                    continue
                    
                labels = extra["y"].to(device=dist_util.dev(), dtype=th.float32)
                batch = batch.to(dist_util.dev())
                
                t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
                
                logits = model(batch, timesteps=t)
                
                loss = loss_fn(th.flatten(logits), labels)
                probs = th.sigmoid(th.flatten(logits)).detach()
                
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                sample_count += batch_size
                
                preds = (probs > args.pred_threshold).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        world_size = dist.get_world_size()
        
        if sample_count > 0:
            avg_loss = total_loss / sample_count
            
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
        else:
            avg_loss = 0.0
            acc = 0.0
            f1 = 0.0
        
        metrics = th.tensor([avg_loss, acc, f1, sample_count], device=dist_util.dev())
        
        gathered_metrics = [th.zeros_like(metrics) for _ in range(world_size)]
        dist.all_gather(gathered_metrics, metrics)
        
        total_samples = sum(m[3].item() for m in gathered_metrics)
        
        if total_samples > 0:
            weighted_loss = sum(m[0].item() * m[3].item() for m in gathered_metrics) / total_samples
            weighted_acc = sum(m[1].item() * m[3].item() for m in gathered_metrics) / total_samples
            weighted_f1 = sum(m[2].item() * m[3].item() for m in gathered_metrics) / total_samples
        else:
            weighted_loss = 0.0
            weighted_acc = 0.0
            weighted_f1 = 0.0
        
        model.train()
        return weighted_loss, weighted_acc, weighted_f1

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        
        loss, acc, f1 = forward_backward_log(train_data)
        mp_trainer.optimize(opt)

        if step % args.log_interval == 0:
            val_loss, val_acc, val_f1 = evaluate_model(model, val_loader)
            
            logger.logkv("train_loss", loss)
            logger.logkv("train_accuracy", acc)
            logger.logkv("train_f1", f1)
            logger.logkv("val_loss", val_loss)
            logger.logkv("val_accuracy", val_acc)
            logger.logkv("val_f1", val_f1)
            logger.dumpkvs()
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                if dist.get_rank() == 0:
                    save_model(mp_trainer, opt, step + resume_step, is_best=True)
                    logger.log(f"New best F1 score: {best_f1:.4f}")
        
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        save_model(mp_trainer, opt, step + resume_step)
    
    dist.barrier()

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, is_best=False):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))
        
        if is_best:
            th.save(
                mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
                os.path.join(logger.get_dir(), "model_best.pt"),
            )


def split_microbatches(microbatch, *args):
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
        iterations=20000,
        lr=1e-4,
        weight_decay=0.05,
        anneal_lr=True,  # bash script uses True
        batch_size=32,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=1000,
        pos_weight=1.0,
        pred_threshold=0.5,
        image_size=64,
        rgb=False,
        in_channels=1,  # FM uses only topology (1 channel)
        output_dim=1,
        val_split=0.2,
        random_seed=42,
        
        # Classifier parameters with attention
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,  # bash script uses 2
        classifier_attention_resolutions="32,16,8",  # FM uses attention
        classifier_use_scale_shift_norm=True,
        classifier_resblock_updown=True,
        classifier_pool="attention",  # bash script uses attention
        
        # Data augmentation
        enable_augmentation=True,  # bash script uses True
        save_augmentation=True,  # augment_data_dir specified in bash script
        augment_data_dir="",  # match bash script parameter name
    )
    
    # Add diffusion defaults which include diffusion_steps, noise_schedule, etc.
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    return parser

if __name__ == "__main__":
    main()