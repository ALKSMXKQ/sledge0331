import os
import copy
import random
import torch
import numpy as np
from PIL import Image
from sledge.diffusion.modelling.ldm_pipeline import LDMPipeline
from sledge.autoencoder.callbacks.rvae_visualization_callback import get_sledge_vector_as_raster


# ==========================================
# 【滤镜函数】Sledgeboard 的核心“美颜”逻辑
# ==========================================
def apply_sledgeboard_filter(vector_data, confidence_threshold=0.5):
    """
    清洗底层的向量数据，过滤掉模型低置信度的“乱线”和“噪点”。
    """
    clean_data = copy.deepcopy(vector_data)
    for key in dir(clean_data):
        if not key.startswith("_"):
            element = getattr(clean_data, key)
            if hasattr(element, 'mask'):
                mask = getattr(element, 'mask')
                # 置信度低于阈值的点，强行抹除不画
                clean_mask = np.where(mask < confidence_threshold, 0.0, mask)
                setattr(element, 'mask', clean_mask)
    return clean_data


# ==========================================
# 【严谨对比函数】底层数学张量的递归比对
# ==========================================
def check_vectors_equal(vec1, vec2):
    """
    遍历两个 SledgeVector 的所有底层 numpy 矩阵，进行逐元素比对。
    """
    for key in dir(vec1):
        if not key.startswith("_"):
            elem1 = getattr(vec1, key)
            elem2 = getattr(vec2, key)

            # 对比 states (坐标张量)
            if hasattr(elem1, 'states') and hasattr(elem2, 'states'):
                if not np.allclose(elem1.states, elem2.states, atol=1e-5):
                    return False
            # 对比 mask (置信度张量)
            if hasattr(elem1, 'mask') and hasattr(elem2, 'mask'):
                if not np.allclose(elem1.mask, elem2.mask, atol=1e-5):
                    return False
    return True


# ==========================================
# 主程序
# ==========================================
def verify_fixed_noise():
    # 您实际的 checkpoint 和图片保存路径
    checkpoint_path = "/home16T/home8T_1/leitingting/sledge_workspace/exp/exp/training_dit_model/training_dit_diffusion/2025.10.17.18.36.55/checkpoint"
    save_dir = "/home16T/home8T_1/leitingting/sledge_workspace/sledge/png/"
    os.makedirs(save_dir, exist_ok=True)

    print(f"正在加载训练好的模型: {checkpoint_path}")
    pipeline = LDMPipeline.from_pretrained(checkpoint_path).to("cuda")

    # 参数设定
    num_classes = 4
    class_labels = [3, 3, 3, 3]  # 一次生成 4 张波士顿地图，完美绕过 batch_size=1 导致的 0-d tensor 挤压 bug
    fixed_seed = 42
    inference_steps = 100  # 必须是 100 步，保证去噪充分
    clean_threshold = 0.5  # 美颜滤镜强度

    saved_raw_vectors = []

    print("\n========== 实验开始：多次生成 ==========")
    for i in range(3):
        print(f"\n--> 正在进行第 {i + 1} 次推理去噪...")

        # ==============================================================
        # 【终极暴力锁】：必须放在 for 循环内部！
        # 每次推理前，强行把系统所有可能的随机引擎全部重置回起点！
        # ==============================================================
        torch.manual_seed(fixed_seed)
        torch.cuda.manual_seed_all(fixed_seed)
        np.random.seed(fixed_seed)
        random.seed(fixed_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        generator = torch.Generator(device=pipeline.device).manual_seed(fixed_seed)

        with torch.no_grad():
            sledge_vectors = pipeline(
                class_labels=class_labels,
                num_inference_timesteps=inference_steps,
                guidance_scale=4.0,
                generator=generator,
                num_classes=num_classes
            )

        # 1. 提取 batch 里的第 1 张地图，并转成 Numpy
        raw_vector_data = sledge_vectors[0].torch_to_numpy()
        saved_raw_vectors.append(raw_vector_data)

        # 2. 调用美颜滤镜，清洗掉低置信度噪点
        clean_vector_data = apply_sledgeboard_filter(raw_vector_data, confidence_threshold=clean_threshold)

        # 3. 转换成像素矩阵
        raster_image = get_sledge_vector_as_raster(clean_vector_data, pipeline.decoder._config)

        # 4. 保存到专门的 png 文件夹
        img = Image.fromarray(raster_image.astype(np.uint8))
        img_name = os.path.join(save_dir, f"verify_scene_clean_run_{i + 1}.png")
        img.save(img_name)
        print(f"    生成完毕！标准可视化图像已保存为: {img_name}")

    print("\n========== 终极数学验证 (底层张量级别) ==========")
    # 直接比较底层浮点数矩阵，不仅肉眼要一样，数学上也要绝对一致
    is_1_2_match = check_vectors_equal(saved_raw_vectors[0], saved_raw_vectors[1])
    is_2_3_match = check_vectors_equal(saved_raw_vectors[1], saved_raw_vectors[2])

    if is_1_2_match and is_2_3_match:
        print("🎉 验证成功！三次生成的底层数学坐标及Mask张量【完全一模一样】！")
        print("这证明 SLEDGE 的生成过程已经被彻底控制！")
    else:
        print("❌ 发现张量差异！底层存在未知的随机泄漏。")


if __name__ == "__main__":
    verify_fixed_noise()