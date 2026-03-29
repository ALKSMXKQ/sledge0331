import torch
import os
import numpy as np
from diffusers import DDIMScheduler  # 必须导入这个
from sledge.diffusion.modelling.ldm_pipeline import LDMPipeline
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint
import hydra
from omegaconf import DictConfig

# ==========================================
# 第一步：在最开头锁死所有硬件随机性
# ==========================================
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 针对 CUDA 原子操作的一致性设置
#torch.use_deterministic_algorithms(True)  # 强制使用确定性算法
#torch.backends.cudnn.deterministic = True  # 锁死 cuDNN
#torch.backends.cudnn.benchmark = False  # 关闭自动算法优化


@hydra.main(config_path="sledge/script/config/diffusion", config_name="default_diffusion")
def main(cfg: DictConfig):
    # 1. 加载预训练好的 Pipeline
    print("\n[SYSTEM] Loading pipeline from checkpoint...")
    pipeline = build_pipeline_from_checkpoint(cfg)

    # ==========================================
    # 第二步：强制替换调度器为 DDIM (确定性模式)
    # ==========================================
    print("[SYSTEM] Replacing DDPMScheduler with DDIMScheduler for 100% determinism...")
    # 从原有的配置中继承 beta_start, beta_end 等参数
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline.to("cuda")

    # 2. 设置实验参数
    seed = 0
    num_samples = 4
    class_labels = [0] * num_samples

    # 3. 运行第一次生成 (Run A)
    print("\n--- Running Generation A (Deterministic) ---")
    generator_a = torch.Generator(device=pipeline.device).manual_seed(seed)
    output_a = pipeline(
        class_labels=class_labels,
        generator=generator_a,
        eta=0.0,  # DDIM 下 eta=0 意味着完全关闭随机噪声采样
        num_inference_timesteps=cfg.num_inference_timesteps
    )

    # 4. 运行第二次生成 (Run B)
    print("--- Running Generation B (Deterministic) ---")
    generator_b = torch.Generator(device=pipeline.device).manual_seed(seed)
    output_b = pipeline(
        class_labels=class_labels,
        generator=generator_b,
        eta=0.0,
        num_inference_timesteps=cfg.num_inference_timesteps
    )

    # 5. 数值一致性检查
    print("\n--- Final Verification Results (DDIM Mode) ---")
    for i in range(num_samples):
        # 校验车辆、行人、地图线的状态张量
        v_diff = torch.abs(output_a[i].vehicles.states - output_b[i].vehicles.states).sum().item()
        p_diff = torch.abs(output_a[i].pedestrians.states - output_b[i].pedestrians.states).sum().item()
        l_diff = torch.abs(output_a[i].lines.states - output_b[i].lines.states).sum().item()

        print(f"Sample {i}:")
        print(f"  - Vehicle Diff: {v_diff:.8f}")
        print(f"  - Pedestrian Diff: {p_diff:.8f}")
        print(f"  - Lines Diff: {l_diff:.8f}")

        # 如果数值完全一致（Diff 为 0），则通过
        if v_diff == 0 and p_diff == 0 and l_diff == 0:
            print(f"  ✅ 结论: 完全一致 (Perfect Match)")
        else:
            print(f"  ❌ 结论: 仍有差异 (Check for other random sources)")


if __name__ == "__main__":
    main()