JOB_NAME=feature_caching
AUTOENCODER_CACHE_PATH=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache
USE_CACHE_WITHOUT_DATASET=False
SEED=0

for MAP in "filter_pgh" "filter_lav" "filter_sgp" "filter_bos" 
do
    python $SLEDGE_DEVKIT_ROOT/sledge/script/run_autoencoder.py \
    py_func=feature_caching \
    seed=$SEED \
    job_name=$JOB_NAME \
    +autoencoder=training_rvae_model \
    scenario_builder=nuplan_test \
    scenario_filter=$MAP \
    cache.autoencoder_cache_path=$AUTOENCODER_CACHE_PATH \
    cache.use_cache_without_dataset=$USE_CACHE_WITHOUT_DATASET \
    callbacks="[]" 
done