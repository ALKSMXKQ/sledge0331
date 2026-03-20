JOB_NAME=training_dit_diffusion
AUTOENCODER_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt
DIFFUSION_CHECKPOINT=null # set for weight intialization / continue training
DIFFUSION_MODEL=dit_b_model # [dit_s_model, dit_b_model, dit_l_model, dit_xl_model]
CLEANUP_DIFFUSION_CACHE=false
SEED=0

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

accelerate launch $SLEDGE_DEVKIT_ROOT/sledge/script/run_diffusion.py \
py_func=training \
seed=$SEED \
job_name=$JOB_NAME \
+diffusion=training_dit_model \
diffusion_model=$DIFFUSION_MODEL \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.cleanup_diffusion_cache=$CLEANUP_DIFFUSION_CACHE \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
diffusion_checkpoint=$DIFFUSION_CHECKPOINT  