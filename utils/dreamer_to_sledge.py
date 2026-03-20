import numpy as np
import pickle
from typing import Dict, Any, Tuple, Optional

# 引入 Sledge 的数据结构 和 配置类
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeVector, SledgeVectorElement, SledgeConfig
)


class DreamerAdapter:
    def __init__(self):
        # 使用默认配置
        self.config = SledgeConfig()

    def _pad_and_mask(self, array: np.ndarray, max_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        将数据填充到固定大小，并生成 mask。
        [NumPy Version]: 返回 Numpy 数组，且不带 Batch 维度 (适配 Simulation)
        """
        curr_num = array.shape[0]
        feature_dims = array.shape[1:]

        # 目标形状: [Max_Num, *Feature_Dims]
        states_shape = (max_num,) + feature_dims

        # 初始化全0
        states = np.zeros(states_shape, dtype=np.float32)
        mask = np.zeros((max_num,), dtype=bool)  # Shape [Max_Num]

        limit = min(curr_num, max_num)
        if limit > 0:
            states[:limit] = array[:limit]
            mask[:limit] = True

        return states, mask

    def _resample_polyline(self, polyline: np.ndarray, target_points: int) -> np.ndarray:
        """简单的重采样 (NumPy)"""
        if len(polyline) < 2:
            return np.zeros((target_points, 2), dtype=np.float32)

        dists = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        cum_dists = np.concatenate([[0], np.cumsum(dists)])
        total_length = cum_dists[-1]

        if total_length == 0:
            return np.repeat(polyline[0:1], target_points, axis=0).astype(np.float32)

        target_dists = np.linspace(0, total_length, target_points)
        x_interp = np.interp(target_dists, cum_dists, polyline[:, 0])
        y_interp = np.interp(target_dists, cum_dists, polyline[:, 1])

        return np.stack([x_interp, y_interp], axis=1).astype(np.float32)

    def load_scenario(self, pkl_path: str) -> SledgeVector:
        """读取 Scenario Dreamer 生成的 pkl 并转换为 SledgeVector (NumPy)"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # --- 1. 智能读取 Agents ---
        if 'agents' in data:
            all_agents = data['agents']
        elif 'agent_states' in data:
            raw = data['agent_states']
            if len(raw.shape) == 2:
                all_agents = raw[:, None, :]  # [N, 1, 8]
            else:
                all_agents = raw
        else:
            if 'agent' in data:
                all_agents = data['agent']
            else:
                raise KeyError(f"文件 {pkl_path} 中找不到 agents 数据。现有键: {list(data.keys())}")

        # --- 2. 智能读取 Agent Types ---
        if 'agent_types' in data:
            agent_types = data['agent_types']
        else:
            num_agents = all_agents.shape[0]
            agent_types = np.zeros((num_agents, 5))
            agent_types[:, 1] = 1

            # --- 3. 提取 Ego 和 Others ---
        if len(all_agents.shape) == 2:
            all_agents = all_agents[:, None, :]

        # 假设最后一个是 Ego
        ego_state_t0 = all_agents[-1, 0, :]
        other_agents_t0 = all_agents[:-1, 0, :]
        other_types = agent_types[:-1]

        # 筛选车辆/行人
        if len(other_types.shape) == 1:
            is_vehicle = other_types == 1
            is_pedestrian = other_types == 2
        else:
            is_vehicle = other_types[:, 1] == 1
            is_pedestrian = other_types[:, 2] == 1

        vehicles_t0 = other_agents_t0[is_vehicle]
        pedestrians_t0 = other_agents_t0[is_pedestrian]

        # --- 4. 坐标转换准备 (Global -> Ego Centric) ---
        ego_x, ego_y = ego_state_t0[0], ego_state_t0[1]
        ego_heading = ego_state_t0[4]

        c, s = np.cos(ego_heading), np.sin(ego_heading)
        R = np.array([[c, -s], [s, c]])
        R_inv = R.T

        def to_ego_frame(points):
            if len(points) == 0: return points
            orig_shape = points.shape
            points_reshaped = points.reshape(-1, 2)
            transformed = (points_reshaped - np.array([ego_x, ego_y])) @ R_inv
            return transformed.reshape(orig_shape)

        def rotate_vector(vecs):
            if len(vecs) == 0: return vecs
            orig_shape = vecs.shape
            vecs_reshaped = vecs.reshape(-1, 2)
            transformed = vecs_reshaped @ R_inv
            return transformed.reshape(orig_shape)

        # --- 5. 构造 Sledge Features ---
        # Ego (vx, vy, ax, ay)
        ego_global_v = ego_state_t0[2:4]  # vx, vy
        ego_local_v = rotate_vector(ego_global_v[None, :])[0]

        # 使用 numpy array
        ego_feature = np.array([[
            ego_local_v[0], ego_local_v[1], 0.0, 0.0
        ]], dtype=np.float32)

        def convert_agent_format(agents_arr):
            if len(agents_arr) == 0: return np.zeros((0, 6), dtype=np.float32)

            pos_global = agents_arr[:, 0:2]
            pos_local = to_ego_frame(pos_global)

            vx_global = agents_arr[:, 2]
            vy_global = agents_arr[:, 3]
            v_scalar = np.sqrt(vx_global ** 2 + vy_global ** 2)

            heading_global = agents_arr[:, 4]
            heading_local = (heading_global - ego_heading + np.pi) % (2 * np.pi) - np.pi

            length = agents_arr[:, 5]
            width = agents_arr[:, 6]

            feat = np.stack([
                pos_local[:, 0], pos_local[:, 1],
                heading_local,
                width, length,
                v_scalar
            ], axis=1)
            return feat.astype(np.float32)

        vec_veh = convert_agent_format(vehicles_t0)
        vec_ped = convert_agent_format(pedestrians_t0)

        # --- 6. 处理车道线 ---
        if 'lanes' in data:
            raw_lanes = data['lanes']
        elif 'road_points' in data:
            raw_lanes = data['road_points']
        else:
            raw_lanes = []

        processed_lanes = []
        for lane in raw_lanes:
            if len(lane) == 0: continue
            lane_local = to_ego_frame(lane[:, :2])
            resampled = self._resample_polyline(lane_local, self.config.num_line_poses)
            processed_lanes.append(resampled)

        if len(processed_lanes) > 0:
            lines_tensor = np.stack(processed_lanes).astype(np.float32)
        else:
            lines_tensor = np.zeros((0, self.config.num_line_poses, 2), dtype=np.float32)

        # --- 7. 组装 ---
        vec_lines_states, vec_lines_mask = self._pad_and_mask(lines_tensor, self.config.num_lines)
        vec_veh_states, vec_veh_mask = self._pad_and_mask(vec_veh, self.config.num_vehicles)
        vec_ped_states, vec_ped_mask = self._pad_and_mask(vec_ped, self.config.num_pedestrians)
        vec_ego_states, vec_ego_mask = self._pad_and_mask(ego_feature, 1)

        empty_static, mask_static = self._pad_and_mask(np.zeros((0, 5), dtype=np.float32),
                                                       self.config.num_static_objects)
        empty_green, mask_green = self._pad_and_mask(np.zeros((0, 2), dtype=np.float32), self.config.num_green_lights)
        empty_red, mask_red = self._pad_and_mask(np.zeros((0, 2), dtype=np.float32), self.config.num_red_lights)

        return SledgeVector(
            lines=SledgeVectorElement(vec_lines_states, vec_lines_mask),
            vehicles=SledgeVectorElement(vec_veh_states, vec_veh_mask),
            pedestrians=SledgeVectorElement(vec_ped_states, vec_ped_mask),
            static_objects=SledgeVectorElement(empty_static, mask_static),
            green_lights=SledgeVectorElement(empty_green, mask_green),
            red_lights=SledgeVectorElement(empty_red, mask_red),
            ego=SledgeVectorElement(vec_ego_states, vec_ego_mask)
        )