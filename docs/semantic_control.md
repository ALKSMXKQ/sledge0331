


后半段流程：
original raw + edited raw
→ sledge_raw_feature_processing
→ encode_raster
→ diff mask + ROI mask
→ half-denoise refinement
→ semantic/compliance 筛选
→ 导出 simulator-ready sledge_vector.gz

python sledge/script/build_multiscenario_raw_cache.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --glob-pattern "**/sledge_raw.gz" \
  --max-scenes 5 \
  --crossing-ratio 0.40 \
  --cut-in-ratio 0.30 \
  --hard-brake-ratio 0.30 \
  --mild-ratio 0.50 \
  --moderate-ratio 0.35 \
  --aggressive-ratio 0.15

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_half_denoise_from_tiered_cache.py \
  --original-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --edited-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_raw_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/multiscenario_refine_output \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt \
  --diffusion-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint \
  --guidance-scale 4.0 \
  --low-noise-start-step-seq 10,12,14 \
  --repair-attempts 6 \
  --save-visuals \
  --save-latents \
  --max-scenes 5