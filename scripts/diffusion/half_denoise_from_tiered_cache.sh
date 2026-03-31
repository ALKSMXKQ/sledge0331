#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=/home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml
AUTOENCODER_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt
DIFFUSION_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint
ORIGINAL_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
EDITED_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/tiered_crossing_raw_cache
OUTPUT_DIR=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/half_denoise_from_tiered_cache_alternating

export CUDA_VISIBLE_DEVICES=1

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_half_denoise_from_tiered_cache.py \
  --original-dir "$ORIGINAL_DIR" \
  --edited-dir "$EDITED_DIR" \
  --output "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --autoencoder-checkpoint "$AUTOENCODER_CHECKPOINT" \
  --diffusion-checkpoint "$DIFFUSION_CHECKPOINT" \
  --num-inference-timesteps 24 \
  --guidance-scale 4.0 \
  --round-start-step-seq 14,10,6 \
  --repair-attempts 4 \
  --alignment-threshold 0.70 \
  --min-preservation-ratio 0.93 \
  --diff-threshold 1e-4 \
  --diff-mask-dilation 3 \
  --roi-mask-dilation 2 \
  --pedestrian-roi-strength 1.0 \
  --roadside-anchor-strength 1.0 \
  --lane-anchor-strength 1.0 \
  --crossing-corridor-strength 0.95 \
  --generic-roi-strength 0.90 \
  --projection-inner-iters 2 \
  --projection-x-alpha 0.35 \
  --projection-y-alpha 0.55 \
  --projection-heading-alpha 0.55 \
  --projection-velocity-alpha 0.55 \
  --projection-size-alpha 0.20 \
  --projection-max-pos-shift-m 1.50 \
  --projection-max-heading-shift-rad 0.70 \
  --projection-max-speed-delta 0.80 \
  --projection-match-max-dist 6.0 \
  --seed 0 \
  --save-visuals \
  --save-latents
