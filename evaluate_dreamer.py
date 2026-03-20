import logging
import os
import hydra
import pandas as pd
import inspect  # 引入反射模块，彻底解决参数名问题
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ==================== 1. 核心引擎与聚合器兼容 ====================
try:
    from nuplan.planning.metrics.metric_engine import MetricEngine
except ImportError:
    from nuplan.planning.metrics.metric_engine import MetricsEngine as MetricEngine

try:
    from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
    from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator
except ImportError:
    from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
    from nuplan.planning.metrics.aggregator.weighted_average_metric_aggregator import WeightedAverageMetricAggregator

# ==================== 2. 指标导入 ====================
try:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import EgoLaneChange
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lane_change import \
        EgoLaneChangeStatistics as EgoLaneChange

try:
    from nuplan.planning.metrics.evaluation_metrics.common.collision_violation import CollisionViolation
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import \
        EgoAtFaultCollisionStatistics as CollisionViolation

try:
    from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_violation import DrivableAreaViolation
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.drivable_area_compliance import \
        DrivableAreaComplianceStatistics as DrivableAreaViolation

try:
    from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_violation import DrivingDirectionViolation
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.driving_direction_compliance import \
        DrivingDirectionComplianceStatistics as DrivingDirectionViolation

try:
    from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_violation import SpeedLimitViolation
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance import \
        SpeedLimitComplianceStatistics as SpeedLimitViolation

try:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import \
        EgoProgressAlongExpertRoute
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_progress_along_expert_route import \
        EgoProgressAlongExpertRouteStatistics as EgoProgressAlongExpertRoute

try:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable import EgoIsComfortableStatistics
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable_statistics import \
        EgoIsComfortableStatistics

try:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk import EgoJerkStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration import EgoLatAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration import EgoLonAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk import EgoLonJerkStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration import EgoYawAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate import EgoYawRateStatistics
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_jerk_statistics import EgoJerkStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lat_acceleration_statistics import \
        EgoLatAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_acceleration_statistics import \
        EgoLonAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_lon_jerk_statistics import EgoLonJerkStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_acceleration_statistics import \
        EgoYawAccelerationStatistics
    from nuplan.planning.metrics.evaluation_metrics.common.ego_yaw_rate_statistics import EgoYawRateStatistics

try:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress import EgoIsMakingProgressStatistics
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.ego_is_making_progress_statistics import \
        EgoIsMakingProgressStatistics

try:
    from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound import \
        TimeToCollisionStatistics
except ImportError:
    from nuplan.planning.metrics.evaluation_metrics.common.time_to_collision_within_bound_statistics import \
        TimeToCollisionStatistics

# ==================== 3. 模拟器组件兼容 ====================
try:
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
except ImportError:
    from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation as TracksObservation

try:
    from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
except ImportError:
    from nuplan.planning.simulation.controller.perfect_tracking.perfect_tracking_controller import \
        PerfectTrackingController

try:
    from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import \
        StepSimulationTimeController
except ImportError:
    from nuplan.planning.simulation.time_controller.step_simulation_time_controller import StepSimulationTimeController

try:
    from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
except ImportError:
    from nuplan.planning.simulation.simulation_runner import SimulationRunner

from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.callback.multi_callback import MultiCallback

# ==================== Sledge Imports ====================
from sledge.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from sledge.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from sledge.simulation.scenarios.sledge_scenario.sledge_scenario import SledgeScenario
from utils.dreamer_to_sledge import DreamerAdapter

# 手动类别映射表
METRIC_CATEGORY_MAP = {
    "ego_lane_change": "dynamic",
    "ego_jerk": "dynamic",
    "ego_lat_acceleration": "dynamic",
    "ego_lon_acceleration": "dynamic",
    "ego_lon_jerk": "dynamic",
    "ego_yaw_acceleration": "dynamic",
    "ego_yaw_rate": "dynamic",
    "no_ego_at_fault_collisions": "safety",
    "time_to_collision_within_bound": "safety",
    "drivable_area_compliance": "compliance",
    "driving_direction_compliance": "compliance",
    "speed_limit_compliance": "compliance",
    "ego_is_comfortable": "comfort",
    "ego_progress_along_expert_route": "planning",
    "ego_is_making_progress": "planning",
}


def build_metrics():
    """构建完整 SLEDGE 评估指标列表"""
    ego_lane_change = EgoLaneChange(name="ego_lane_change", category="dynamic", max_fail_rate=0.2)

    ego_jerk = EgoJerkStatistics(name="ego_jerk", category="dynamic", max_abs_mag_jerk=10.0)
    ego_lat_accel = EgoLatAccelerationStatistics(name="ego_lat_acceleration", category="dynamic", max_abs_lat_accel=5.0)
    ego_lon_accel = EgoLonAccelerationStatistics(name="ego_lon_acceleration", category="dynamic", min_lon_accel=-6.0,
                                                 max_lon_accel=4.0)
    ego_lon_jerk = EgoLonJerkStatistics(name="ego_lon_jerk", category="dynamic", max_abs_lon_jerk=10.0)
    ego_yaw_accel = EgoYawAccelerationStatistics(name="ego_yaw_acceleration", category="dynamic", max_abs_yaw_accel=3.0)
    ego_yaw_rate = EgoYawRateStatistics(name="ego_yaw_rate", category="dynamic", max_abs_yaw_rate=1.0)

    collision_metric = CollisionViolation(name="no_ego_at_fault_collisions", category="safety",
                                          ego_lane_change_metric=ego_lane_change)
    ego_progress = EgoProgressAlongExpertRoute(name="ego_progress_along_expert_route", category="planning")

    comfort_metric = EgoIsComfortableStatistics(
        name="ego_is_comfortable", category="comfort",
        ego_jerk_metric=ego_jerk, ego_lat_acceleration_metric=ego_lat_accel,
        ego_lon_acceleration_metric=ego_lon_accel, ego_lon_jerk_metric=ego_lon_jerk,
        ego_yaw_acceleration_metric=ego_yaw_accel, ego_yaw_rate_metric=ego_yaw_rate
    )

    ttc_metric = TimeToCollisionStatistics(
        name="time_to_collision_within_bound", category="safety",
        ego_lane_change_metric=ego_lane_change, no_ego_at_fault_collisions_metric=collision_metric,
        time_step_size=0.1, time_horizon=5.0, least_min_ttc=0.0
    )

    making_progress_metric = EgoIsMakingProgressStatistics(
        name="ego_is_making_progress", category="planning",
        ego_progress_along_expert_route_metric=ego_progress, min_progress_threshold=2.0
    )

    return [
        ego_lane_change, ego_jerk, ego_lat_accel, ego_lon_accel, ego_lon_jerk, ego_yaw_accel, ego_yaw_rate,
        collision_metric, ego_progress, ttc_metric, comfort_metric, making_progress_metric,
        DrivableAreaViolation(name="drivable_area_compliance", category="compliance",
                              lane_change_metric=ego_lane_change, max_violation_threshold=0.5),
        DrivingDirectionViolation(name="driving_direction_compliance", category="compliance",
                                  lane_change_metric=ego_lane_change),
        SpeedLimitViolation(name="speed_limit_compliance", category="compliance", lane_change_metric=ego_lane_change,
                            max_violation_threshold=3, max_overspeed_value_threshold=1.0),
    ]


def safe_getattr(obj, attr_list, default=None):
    """安全获取对象属性"""
    for attr in attr_list:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default


def smart_create_metric_dataframe(df, metric_name):
    """
    【万能适配函数】智能实例化 MetricStatisticsDataFrame
    根据类的实际签名自动决定传入哪些参数
    """
    try:
        # 获取类的签名
        sig = inspect.signature(MetricStatisticsDataFrame)
        required_params = [p for name, p in sig.parameters.items()
                           if name != 'self' and p.default == inspect.Parameter.empty]

        # 策略 1: 如果只需要 1 个参数，直接传 df
        if len(required_params) == 1:
            return MetricStatisticsDataFrame(df)

        # 策略 2: 如果需要 2 个参数，猜测是 (name, df)
        elif len(required_params) == 2:
            print(f"Debug: 检测到需要双参数 {required_params}, 尝试传入 (name, df)")
            return MetricStatisticsDataFrame(metric_name, df)

        # 策略 3: 参数多于2个或无法判断，尝试关键字匹配
        else:
            kwargs = {}
            for name, param in sig.parameters.items():
                if name == 'self': continue
                # 模糊匹配参数名
                if 'dataframe' in name or 'statistics' in name:
                    kwargs[name] = df
                elif 'name' in name or 'key' in name:
                    kwargs[name] = metric_name
                elif param.default == inspect.Parameter.empty:
                    kwargs[name] = None  # 无法猜测的必选参数，填None碰运气

            return MetricStatisticsDataFrame(**kwargs)

    except Exception as e:
        print(f"⚠️ 智能实例化失败，降级尝试位置参数: {e}")
        try:
            return MetricStatisticsDataFrame(df)
        except:
            return MetricStatisticsDataFrame(metric_name, df)


@hydra.main(config_path="sledge/script/config/simulation", config_name="default_simulation")
def main(cfg):
    dreamer_output_dir = "/home16T/home8T_1/leitingting/scenario-dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/complete_sim_envs"

    if not os.path.exists(dreamer_output_dir):
        print(f"Error: 路径不存在 -> {dreamer_output_dir}")
        return

    files = [os.path.join(dreamer_output_dir, f) for f in os.listdir(dreamer_output_dir) if f.endswith('.pkl')]
    if not files:
        print(f"Warning: 在 {dreamer_output_dir} 中没有找到 .pkl 文件。")
        return

    adapter = DreamerAdapter()
    output_dir = Path.cwd() / "simulation_results"
    log_dir = output_dir / "simulation_log"
    metric_dir = output_dir / "metrics"
    log_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始评估: 共找到 {len(files)} 个场景文件。")

    metric_engine = MetricEngine(metric_dir, build_metrics())
    metric_files = []

    for pkl_file in tqdm(files):
        try:
            sledge_vector = adapter.load_scenario(pkl_file)
            scenario = SledgeScenario(Path(pkl_file), sledge_vector=sledge_vector)

            idm_policy = BatchIDMPolicy(speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0], fallback_target_velocity=15.0,
                                        min_gap_to_lead_agent=1.0, headway_time=1.5, accel_max=1.5, decel_max=3.0)
            planner = PDMClosedPlanner(trajectory_sampling=scenario._future_sampling,
                                       proposal_sampling=scenario._future_sampling, idm_policies=idm_policy,
                                       lateral_offsets=[0.0], map_radius=100.0)

            observations = TracksObservation(scenario=scenario)
            ego_controller = PerfectTrackingController(scenario)
            simulation_time_controller = StepSimulationTimeController(scenario)
            simulation_setup = SimulationSetup(time_controller=simulation_time_controller, observations=observations,
                                               ego_controller=ego_controller, scenario=scenario)

            log_callback = SimulationLogCallback(output_directory=log_dir, simulation_log_dir=log_dir,
                                                 serialization_type="msgpack")
            simulation = Simulation(simulation_setup=simulation_setup, callback=MultiCallback([log_callback]),
                                    simulation_history_buffer_duration=cfg.get("simulation_history_buffer_duration",
                                                                               2.0))

            runner = SimulationRunner(simulation, planner)
            runner_report = runner.run()

            if runner_report.succeeded:
                history = simulation.history
                metrics_dict = metric_engine.compute(history, scenario, planner_name="PDMClosedPlanner")
                for file_list in metrics_dict.values():
                    metric_files.extend(file_list)

        except Exception as e:
            print(f"\nFailed to run {os.path.basename(pkl_file)}: {e}")

    # ================= 6. 手动构建 DataFrame 并聚合 (智能适配版) =================
    if metric_files:
        print(f"\n正在聚合 {len(metric_files)} 个指标文件...")

        try:
            metric_file_dict = defaultdict(list)
            for mf in metric_files:
                metric_file_dict[mf.key.metric_name].append(mf)

            metric_dataframes = {}
            for metric_name, files in metric_file_dict.items():
                data_list = []
                for mf in files:
                    # 遍历 Statistics 列表 (修复层级问题)
                    for result in mf.metric_statistics:

                        # 1. 提取 Result 对象属性
                        row = {
                            "log_name": mf.key.log_name,
                            "scenario_name": mf.key.scenario_name,
                            "scenario_type": mf.key.scenario_type,
                            "planner_name": mf.key.planner_name,
                            "metric_name": mf.key.metric_name,
                            "metric_category": getattr(result, 'metric_category',
                                                       METRIC_CATEGORY_MAP.get(mf.key.metric_name, "unknown")),
                            "metric_score": getattr(result, 'metric_score', None),
                            "metric_score_unit": getattr(result, 'metric_score_unit', None),
                        }

                        # 2. 提取详细统计值
                        if hasattr(result, 'statistics') and result.statistics:
                            for sub_stat in result.statistics:
                                name = safe_getattr(sub_stat, ['name'], 'unknown')
                                val = safe_getattr(sub_stat, ['value', 'stat_value', 'result', 'number'], None)
                                unit = safe_getattr(sub_stat, ['unit', 'stat_unit'], '')
                                typ = safe_getattr(sub_stat, ['type', 'stat_type'], 'unknown')

                                row[f"{name}_stat_value"] = val
                                row[f"{name}_stat_type"] = typ
                                row[f"{name}_stat_unit"] = unit

                        data_list.append(row)

                df = pd.DataFrame(data_list)
                # 【关键调用】使用智能函数实例化
                metric_dataframes[metric_name] = smart_create_metric_dataframe(df, metric_name)

            aggregator = WeightedAverageMetricAggregator(
                name="sledge_weighted_average", metric_weights={}, file_name="metrics",
                aggregator_save_path=metric_dir, multiple_metrics=[]
            )
            aggregator(metric_dataframes)
            print(f"✅ 指标聚合完成！文件已保存至: {metric_dir / 'metrics.parquet'}")

        except Exception as e:
            print(f"❌ 聚合过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ 没有生成任何指标数据。")


if __name__ == "__main__":
    main()