# SLEDGE 语义控制补丁

这个补丁按 **SLEDGE v0.1** 的目录结构组织，核心思路是：

1. 在 `sledge_raw.gz` 进入 `sledge_raw_feature_processing(...)` 之前，先做自然语言解析和实体级编辑。
2. 继续走原始 SLEDGE 流程，把编辑后的场景 rasterize 成 RSI，再用 RVAE encoder 得到 latent。
3. 不从纯噪声开始，而是把 latent 加噪到指定 step（默认 30），然后从该 step 开始反推。
4. 用 latent mask 在每一步把“遮挡物 + 行人”的局部区域重新写回 source latent，尽量保住你人工设定的位置。
5. 最后输出 `denoised_sledge_vector.gz`，并附带 prompt alignment 打分，便于后筛选。

## 文件放置位置

把下面这些文件覆盖/加入到你的工程里：

- `sledge/diffusion/modelling/ldm_pipeline.py`  ← **修改过的原文件**
- `sledge/semantic_control/__init__.py`
- `sledge/semantic_control/io.py`
- `sledge/semantic_control/prompt_spec.py`
- `sledge/semantic_control/prompt_parser.py`
- `sledge/semantic_control/vector_editor.py`
- `sledge/semantic_control/prompt_alignment.py`
- `sledge/script/run_semantic_img2img.py`


- 'sledge/autoencoder/modeling/models/rvae/rvae_decoder.py'
- /home16T/home8T_1/leitingting/sledge_workspace/sledge/sledge/sledgeboard/tabs/overview_tab.py

## 运行示例

```bash
python sledge/script/run_semantic_img2img.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/semantic_batch_out \
  --prompt "创建鬼探头场景：自车前方道路旁有遮挡物，遮挡物后有行人从盲区突然穿出" \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt \
  --diffusion-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint \
  --start-timestep-index 30 \
  --num-inference-timesteps 50 \
  --guidance-scale 4.0 \
  --skip-existing \
  --max-scenes 10
  
python sledge/script/run_semantic_img2img.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/semantic_batch_out \
  --prompt "创建鬼探头场景：自车前方道路旁有遮挡物，遮挡物后有行人从盲区突然穿出" \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt \
  --diffusion-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint \
  --start-timestep-index 30 \
  --num-inference-timesteps 50 \
  --guidance-scale 4.0 \
  --max-scenes 10 \
  --resample-attempts 3 \
  --alignment-threshold 0.7
  
  python -u sledge/script/run_semantic_img2img.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/semantic_batch_out \
  --prompt "创建鬼探头场景：自车前方道路旁有遮挡物，遮挡物后有行人从盲区突然穿出" \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --autoencoder-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_rvae_model/training_rvae_model/2025.10.17.06.17.03/best_model/epoch45.ckpt \
  --diffusion-checkpoint /home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint \
  --start-timestep-index 12 \
  --num-inference-timesteps 24 \
  --guidance-scale 3.0 \
  --mask-dilation 3 \
  --resample-attempts 3 \
  --alignment-threshold 0.7 \
  --max-scenes 10
  

  
  python -u sledge/script/build_edited_scenario_cache.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output /home16T/home8T_1/leitingting/sledge_workspace/exp/edited_eval_out \
  --prompt "中度 突发的行人横穿马路" \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --max-scenes 10
  
  python -u sledge/script/build_tiered_crossing_raw_cache.py \
  --input-dir /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/autoencoder_cache \
  --output-root /home16T/home8T_1/leitingting/sledge_workspace/exp/caches/tiered_crossing_raw_cache \
  --config /home16T/home8T_1/leitingting/sledge_workspace/semantic_img2img_cfg.yaml \
  --base-prompt "突发的行人横穿马路" \
  --mild-ratio 0.50 \
  --moderate-ratio 0.35 \
  --aggressive-ratio 0.15 \
  --skip-existing
  
  python $SLEDGE_DEVKIT_ROOT/sledge/script/run_simulation.py   +simulation=sledge_reactive_agents   planner=pdm_closed_planner   observation=sledge_agents_observation   scenario_builder=nuplan   cache.scenario_cache_path=/home16T/home8T_1/leitingting/sledge_workspace/exp/caches/scenario_cache_half_denoise_best

```

## 输出内容

- `edited_sledge_raw.gz`：自然语言编辑后的原始向量场景
- `denoised_sledge_vector.gz`：img2img 去噪后的最终向量场景
- `prompt_spec.json`：正则化后的 prompt 结构化结果
- `edit_report.json`：插入/删除/减速了哪些实体
- `prompt_alignment.json`：语义满足度评分
- `edited_raster.png` / `denoised_vector.png`：可视化结果
- `init_latents.pt` / `final_latents.pt` / `preserve_mask.pt`：调试文件（可选）

## 当前实现范围

这一版优先把你最明确提出的 **“鬼探头 / 遮挡物后行人突然冲出”** 路线打通。
解析器是可扩展的，但目前最稳的是：

- 鬼探头
- 遮挡物类型（车辆 / 静态物）
- 左右侧
- 中等交通
- yielding / 让行
- 盲区 / 遮挡语义

如果你后面还要继续扩成：

- 左转横穿来车
- 路口礼让
- 施工区变道
- 非机动车横穿

可以继续在 `prompt_parser.py` + `vector_editor.py` 上加规则模板。
