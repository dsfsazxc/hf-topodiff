"""
Sample from diffusion model using noisy image regressors and reward models
to guide the sampling process towards more realistic topologies.
"""

import argparse
import os
import random
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from topodiff.cons_input_datasets import load_data
from topodiff import dist_util, logger
from topodiff.script_util import (
    model_and_diffusion_defaults,
    regressor_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_regressor,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def setup_reward_model(args):
    """Setup reward model (time-dependent or time-independent)"""
    kwargs = args_to_dict(args, classifier_defaults().keys())
    model = create_classifier(in_channels=args.reward_in_channels, **kwargs)
    in_ch = model.out[2].c_proj.in_channels
    model.out[2].c_proj = th.nn.Conv1d(in_ch, 1, kernel_size=1)
    return model


def main():
    args = create_argparser().parse_args()

    # Set random seed for reproducibility
    if args.fix_random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        th.manual_seed(args.random_seed)
        th.cuda.manual_seed_all(args.random_seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("Loading regressor...")
    regressor = create_regressor(
        regressor_depth=args.regressor_depth, 
        in_channels=args.regressor_channels, 
        **args_to_dict(args, regressor_defaults().keys())
    )
    regressor.load_state_dict(
        dist_util.load_state_dict(args.regressor_path, map_location="cpu")
    )
    regressor.to(dist_util.dev())
    if args.regressor_use_fp16:
        regressor.convert_to_fp16()
    regressor.eval()

    logger.log("Loading FM classifier...")
    fm_classifier = create_classifier(
        in_channels=1, 
        **args_to_dict(args, classifier_defaults().keys())
    )
    fm_classifier.load_state_dict(
        dist_util.load_state_dict(args.fm_classifier_path, map_location="cpu")
    )
    fm_classifier.to(dist_util.dev())
    if args.fm_classifier_use_fp16:
        fm_classifier.convert_to_fp16()
    fm_classifier.eval()
    
    # Load FM reward models (ensemble)
    fm_reward_list = []
    if args.fm_reward_paths is not None:
        logger.log(f"Loading {len(args.fm_reward_paths)} FM reward models...")
        for i, fm_reward_path in enumerate(args.fm_reward_paths):
            logger.log(f"Loading FM reward model {i+1}: {fm_reward_path}")
            fm_reward = setup_reward_model(args)
            fm_reward.load_state_dict(dist_util.load_state_dict(fm_reward_path, map_location="cpu"))
            fm_reward.to(dist_util.dev())
            if args.reward_use_fp16:
                fm_reward.convert_to_fp16()
            fm_reward.eval()
            fm_reward_list.append(fm_reward)
        logger.log(f"Successfully loaded {len(fm_reward_list)} FM reward models")
    
    # Load BC reward models (ensemble)
    bc_reward_list = []
    if args.bc_reward_paths is not None:
        logger.log(f"Loading {len(args.bc_reward_paths)} BC reward models...")
        for i, bc_reward_path in enumerate(args.bc_reward_paths):
            logger.log(f"Loading BC reward model {i+1}: {bc_reward_path}")
            try:
                bc_reward = create_regressor(
                    regressor_depth=args.regressor_depth, 
                    in_channels=args.regressor_channels, 
                    **args_to_dict(args, regressor_defaults().keys())
                )
                bc_reward.load_state_dict(dist_util.load_state_dict(bc_reward_path, map_location="cpu"))
                bc_reward.to(dist_util.dev())
                if args.regressor_use_fp16:
                    bc_reward.convert_to_fp16()
                bc_reward.eval()
                bc_reward_list.append(bc_reward)
            except Exception as e:
                logger.log(f"Failed to load BC reward model {i+1}: {e}")
        logger.log(f"Successfully loaded {len(bc_reward_list)} BC reward models")

    data = load_data(data_dir=args.constraints_path)

    def cond_fn_1(x, t):
        """Condition function for regressor and BC reward guidance"""
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            
            # Regressor gradient
            logits = regressor(x_in, t)
            reg_grad = th.autograd.grad(logits.sum(), x_in)[0]
            reg_grad_reshaped = reg_grad[:, 0, :, :].reshape(
                (x_in.size(0), 1, args.image_size, args.image_size)
            )
            
            # BC reward ensemble gradient
            bc_reward_grad_final = None
            if len(bc_reward_list) > 0:
                log_p_valid = 0.0
                for bc_reward in bc_reward_list:
                    r = bc_reward(x_in, t)
                    log_p_valid = log_p_valid + F.logsigmoid(r)
                bc_reward_grad = th.autograd.grad(log_p_valid.sum(), x_in)[0]
                bc_reward_grad_final = bc_reward_grad[:, 0, :, :].reshape(
                    (x_in.size(0), 1, args.image_size, args.image_size)
                )
            
            # Combine gradients
            result = (-1) * reg_grad_reshaped * args.regressor_scale
            if bc_reward_grad_final is not None:
                result = result - bc_reward_grad_final * args.bc_reward_scale
                
            return result

    def cond_fn_2(x, t):
        """Condition function for FM classifier and FM reward guidance"""
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)

            # FM Classifier gradient (target class=1)
            y = th.ones(x_in.size(0), device=x_in.device, dtype=th.long)
            logits = fm_classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y]
            fm_cls_grad = th.autograd.grad(selected.sum(), x_in)[0]
            fm_cls_grad_reshaped = fm_cls_grad[:, 0, :, :].reshape(
                (x_in.size(0), 1, args.image_size, args.image_size)
            )

            # FM reward ensemble gradient
            fm_reward_grad_final = None
            if len(fm_reward_list) > 0:
                log_p_valid = 0.0
                for fm_reward in fm_reward_list:
                    r = fm_reward(x_in, t)
                    log_p_valid = log_p_valid + F.logsigmoid(r)
                fm_reward_grad = th.autograd.grad(log_p_valid.sum(), x_in)[0]
                fm_reward_grad_final = fm_reward_grad[:, 0, :, :].reshape(
                    (x_in.size(0), 1, args.image_size, args.image_size)
                )

            # Combine gradients
            result = fm_cls_grad_reshaped * args.classifier_fm_scale
            if fm_reward_grad_final is not None:
                result = result + fm_reward_grad_final * args.fm_reward_scale
            return result

    def model_fn(x, t):
        return model(x, t)

    logger.log("Sampling...")
    all_images = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        input_cons, input_raw_loads, input_raw_BCs = next(data)
        input_cons = input_cons.cuda()
        input_raw_loads = input_raw_loads.cuda()
        input_raw_BCs = input_raw_BCs.cuda()

        sample = sample_fn(
            model_fn,
            (args.batch_size, 1, args.image_size, args.image_size),
            input_cons,
            input_raw_loads,
            input_raw_BCs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn_1=cond_fn_1,
            cond_fn_2=cond_fn_2,
            device=dist_util.dev(),
        )
        
        sample = (sample * 255).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"Created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    if dist.get_rank() == 0:
        # Save samples
        out_path = args.output_file if args.output_file else os.path.join(logger.get_dir(), f"samples.npz")
        logger.log(f"Saving to {out_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("Sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=20,
        batch_size=1,
        use_ddim=False,
        model_path="",
        regressor_path="",
        fm_classifier_path="",
        regressor_scale=1.0,
        classifier_fm_scale=1.0,
        constraints_path="",
        classifier_use_fp16=True,
        regressor_use_fp16=False,
        fm_classifier_use_fp16=False,
        bc_reward_scale=1.0,
        fm_reward_scale=1.0,
        fix_random_seed=False,
        random_seed=42,
        output_file="",
        reward_in_channels=1,
        reward_use_fp16=False,
        regressor_depth=4,
        regressor_channels=8,  # 1 topology + 7 constraint channels
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(regressor_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add reward model paths arguments for ensemble
    parser.add_argument("--fm_reward_paths", type=str, nargs='+', 
                        help="List of paths to FM reward models for ensemble")
    parser.add_argument("--bc_reward_paths", type=str, nargs='+',
                        help="List of paths to BC reward models for ensemble")
    return parser


if __name__ == "__main__":
    main()