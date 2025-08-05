# Censored Sampling for Topology Design: Guiding Diffusion with Human Preferences

**Enhance TopoDiff sampling with simple human feedback.**

* **BC & FM classifiers**: Train minimal binary models for boundary and floating defects
* **Inference guidance**: Apply reward gradients to pretrained TopoDiff (no retraining)

This project builds upon the following repositories:

* [diffusion-human-feedback](https://github.com/tetrzim/diffusion-human-feedback) by tetrzim
* [TopoDiff](https://github.com/francoismaze/topodiff) by francoismaze

## Contents

1. **Installation**
2. **Human Feedback Collection**
3. **Reward Model Training**
4. **Censored Sampling**
---
## 1. Installation

Install the package and dependencies:
```bash
pip install -e .
```
---
## 2. Human Feedback Collection

Use the GUI to annotate generated topology samples:

```bash
python scripts/hf/Topo_feedback_data_collector.py \
    --data_dir1 <path>/sample_imgs_1 \
    --data_dir2 <path>/sample_imgs_2 \
    --data_dir3 <path>/sample_imgs_3 \
    --data_dir4 <path>/sample_imgs_4 \
    --BC_dir <path>/test_data_summary.npy \
    --constraint_data_path <path>/dataset \
    --feedback_output_dir <path>/feedback_dataset \
    --resolution 180 \
    --grid_row 3 \
    --grid_col 4
```

Collected feedback saved as `feedback_bc.npy` and `feedback_fm.npy`.

---

## 3. Reward Model Training

Detailed training scripts with full configuration flags:

**FM Reward Model Training**

```bash

# Train Floating Material (FM) reward model

# Model settings
REWARD_FLAGS="--image_size 64 \
               --in_channels 1 \
               --classifier_attention_resolutions 32,16,8 \
               --classifier_depth 2 \
               --classifier_width 128 \
               --classifier_pool attention \
               --classifier_resblock_updown True \
               --classifier_use_scale_shift_norm True \
               --output_dim 1"

# Diffusion & training settings
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--iterations 20000 \
             --anneal_lr True \
             --lr 1e-4 \
             --batch_size 32 \
             --save_interval 1000 \
             --weight_decay 0.05 \
             --log_interval 10 \
             --val_split 0.2"
AUGMENT_FLAGS="--enable_augmentation True"
CLASS_FLAGS="--pos_weight 1.0 --output_dim 1"

# Paths
DATA_DIR="<path>/feedback_dataset"
AUGMENT_DIR="<path>/logs/fm/augmented_data"
LOG_DIR="<path>/logs/fm"

# Execute
python scripts/reward/FM_reward_train.py \
    --data_dir=$DATA_DIR \
    --augment_data_dir=$AUGMENT_DIR \
    --log_dir=$LOG_DIR \
    $REWARD_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $CLASS_FLAGS $AUGMENT_FLAGS
```

**BC Reward Model Training**

```bash

# Train Boundary Condition (BC) reward model

# Model settings
MODEL_FLAGS="--image_size 64 \
              --in_channels 8 \
              --classifier_width 128 \
              --classifier_depth 4 \
              --classifier_attention_resolutions 32,16,8 \
              --classifier_resblock_updown True \
              --classifier_use_scale_shift_norm True \
              --classifier_pool spatial"

# Diffusion & training settings
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"
TRAIN_FLAGS="--iterations 40000 \
             --anneal_lr False \
             --lr 5e-5 \
             --batch_size 8 \
             --microbatch 2 \
             --save_interval 1000 \
             --weight_decay 0.1 \
             --log_interval 10 \
             --val_split 0.2"
CLASS_FLAGS="--pos_weight 1.0 --output_dim 1"
AUGMENT_FLAGS="--enable_augmentation True"

# Paths
DATA_DIR="<path>/feedback_dataset"
AUGMENT_DIR="<path>/logs/bc/augmented_data"
LOG_DIR="<path>/logs/bc"

# Execute
python scripts/reward/BC_reward_train.py \
    --data_dir $DATA_DIR \
    --augment_data_dir $AUGMENT_DIR \
    --log_dir $LOG_DIR \
    $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $CLASS_FLAGS $AUGMENT_FLAGS
```
---

## 4. Censored Sampling

Generate new topology designs guided by the trained reward models:

```bash
python scripts/censored_topodiff_sample.py \
  --regressor_scale 4.0 --classifier_fm_scale 3.0 \
  --image_size 64 --num_channels 128 --num_res_blocks 3 \
  --learn_sigma True --dropout 0.3 --use_fp16 False \
  --diffusion_steps 1000 --timestep_respacing 100 --noise_schedule cosine \
  --constraints_path ./data/data/dataset_1_diff/test_data_level_1 \
  --num_samples 1 \
  --model_path ./data/data/checkpoints/diff_checkpoints/model_180000.pt \
  --regressor_path ./data/data/checkpoints/reg_checkpoint/model_350000.pt \
  --fm_classifier_path ./data/data/checkpoints/class_checkpoint/model_299999.pt \
  --bc_reward_path <path>/bc_model.pt --lv_reward_scale 2.0 \
  --fm_reward_paths <path>/fm_model.pt --red_reward_scale 2.0 \
  --output_file <path>/samples.npz
```

---
