"""SEM-Trans 最小使用示例。"""

from typing import Dict

from utils.sem_trans import SEMTransInterceptor


def intercept_feature_dict_before_rsi(features: Dict[str, object], instruction: str) -> Dict[str, object]:
    """
    在不修改 SLEDGE 核心源码的前提下，把 SEM-Trans 当成一个外挂式前置拦截器。

    典型调用时机:
        features["sledge_raw"] 已经从日志/Scenario 构建出来，
        但还没有进入 sledge_raw_feature_processing(...) 做 RSI/RLM。
    """
    interceptor = SEMTransInterceptor()
    return interceptor.intercept_features_dict(features, instruction)


# 如果你手里直接拿的是 nuPlan Scenario，也可以这样调:
#
# from utils.sem_trans import SEMTransInterceptor
# interceptor = SEMTransInterceptor()
# result = interceptor.intercept_scenario(
#     scenario=my_scenario,
#     instruction="复杂路口，车流增加，货车后有鬼探头",
#     return_debug=True,
# )
# modified_raw_scene = result.raw_scene
# parsed_intent_json = result.intent.to_json_dict()
# anchor_point = result.anchor.point
