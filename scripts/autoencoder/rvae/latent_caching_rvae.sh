JOB_NAME=latent_caching
AUTOENCODER_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
AUTOENCODER_CHECKPOINT=/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt
USE_CACHE_WITHOUT_DATASET=False
SEED=0

python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
py_func=latent_caching \
seed=$SEED \
job_name=$JOB_NAME \
+autoencoder=training_rvae_model \
data_augmentation=rvae_no_augmentation \
autoencoder_checkpoint=$AUTOENCODER_CHECKPOINT \
cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
cache.latent_name="rvae_latent" \
cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
callbacks="[]" 