import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from bokeh.document.document import Document
from bokeh.models import ColumnDataSource, DataTable, TableColumn

from sledge.sledgeboard.base.base_tab import BaseTab
from sledge.sledgeboard.base.experiment_file_data import ExperimentFileData
from sledge.sledgeboard.tabs.config.overview_tab_config import (
    OVERVIEW_PLANNER_CHECKBOX_GROUP_NAME,
    OverviewAggregatorData,
    OverviewTabDataTableConfig,
    OverviewTabDefaultDataSourceDictConfig,
    OverviewTabExperimentTableColumnConfig,
    OverviewTabPlannerTableColumnConfig,
    OverviewTabScenarioTypeTableColumnConfig,
)

logger = logging.getLogger(__name__)


class OverviewTab(BaseTab):
    """Overview tab in SledgeBoard."""

    def __init__(self, doc: Document, experiment_file_data: ExperimentFileData):
        super().__init__(doc=doc, experiment_file_data=experiment_file_data)

        self._aggregator_metric_data: Dict[str, List[OverviewAggregatorData]] = {}

        self._default_datasource_dict = dict(**OverviewTabDefaultDataSourceDictConfig.get_config())
        self._default_datasource = ColumnDataSource(data=self._default_datasource_dict)

        self._default_columns = [
            TableColumn(**OverviewTabExperimentTableColumnConfig.get_config()),
            TableColumn(**OverviewTabScenarioTypeTableColumnConfig.get_config()),
            TableColumn(**OverviewTabPlannerTableColumnConfig.get_config()),
        ]

        self.table = DataTable(
            source=self._default_datasource,
            columns=self._default_columns,
            **OverviewTabDataTableConfig.get_config(),
        )

        self.planner_checkbox_group.name = OVERVIEW_PLANNER_CHECKBOX_GROUP_NAME

    def file_paths_on_change(
        self,
        experiment_file_data: ExperimentFileData,
        experiment_file_active_index: List[int],
    ) -> None:
        self._experiment_file_data = experiment_file_data
        self._experiment_file_active_index = experiment_file_active_index
        self._overview_on_change()

    def _click_planner_checkbox_group(self, attr: Any) -> None:
        self._update_overview_table(self._aggregator_metric_data)

    def _overview_on_change(self) -> None:
        self._aggregator_metric_data = self._aggregate_metric_aggregator()
        self._update_overview_table(data=self._aggregator_metric_data)

    def _aggregate_metric_aggregator(self) -> Dict[str, List[OverviewAggregatorData]]:
        """
        Aggregate metric aggregator data.
        Filters rows with invalid num_scenarios / score values so Bokeh won't fail on NaN/Inf.
        """
        data: Dict[str, List[OverviewAggregatorData]] = defaultdict(list)
        planner_names: List[str] = []

        for index, metric_aggregator_dataframes in enumerate(self.experiment_file_data.metric_aggregator_dataframes):
            if index not in self._experiment_file_active_index:
                continue

            for metric_aggregator_filename, metric_aggregator_dataframe in metric_aggregator_dataframes.items():
                for _, row_data in metric_aggregator_dataframe.iterrows():
                    num_scenarios = row_data.get("num_scenarios", None)
                    if num_scenarios is None:
                        continue
                    try:
                        if np.isnan(num_scenarios):
                            continue
                    except TypeError:
                        pass
                    if not num_scenarios:
                        continue

                    metric_score = row_data.get("score", None)
                    if metric_score is None:
                        continue
                    try:
                        metric_score = float(metric_score)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(metric_score):
                        continue

                    aggregator_type = row_data.get("aggregator_type", "unknown")
                    planner_name = row_data.get("planner_name", "unknown")
                    scenario_type = row_data.get("scenario_type", "unknown")

                    data[metric_aggregator_filename].append(
                        OverviewAggregatorData(
                            aggregator_type=aggregator_type,
                            planner_name=planner_name,
                            scenario_type=scenario_type,
                            num_scenarios=int(num_scenarios),
                            score=metric_score,
                            aggregator_file_name=metric_aggregator_filename,
                        )
                    )
                    planner_names.append(planner_name)

        sorted_planner_names = sorted(list(set(planner_names)))
        self.planner_checkbox_group.labels = sorted_planner_names
        self.planner_checkbox_group.active = [index for index in range(len(sorted_planner_names))]
        return data

    def _update_overview_table(self, data: Dict[str, List[OverviewAggregatorData]]) -> None:
        """Update overview table with the new metric aggregator data."""
        planner_names = sorted(
            list(
                {
                    metric_aggregator_data.planner_name
                    for _, metric_aggregator_data_list in data.items()
                    for metric_aggregator_data in metric_aggregator_data_list
                    if metric_aggregator_data.planner_name in self.enable_planner_names
                }
            )
        )

        planner_name_columns: Dict[str, List[Any]] = {planner_name: [] for planner_name in planner_names}
        metric_aggregator_files: List[str] = []
        scenario_types: List[str] = []

        for metric_file, metric_aggregator_data_list in data.items():
            metric_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
            initial_planner_names: List[str] = []

            for metric_aggregator_data in metric_aggregator_data_list:
                initial_planner_names.append(metric_aggregator_data.planner_name)
                if metric_aggregator_data.planner_name not in self.enable_planner_names:
                    continue

                score = float(metric_aggregator_data.score)
                if not np.isfinite(score):
                    continue

                metric_scores[metric_aggregator_data.planner_name][metric_aggregator_data.scenario_type] = [
                    np.round(score, 4),
                    metric_aggregator_data.num_scenarios,
                ]

            initial_planner_names = list(set(initial_planner_names))
            metric_aggregator_file_scenario_types: List[str] = []
            metric_file_rows = 0

            for planner_name, values in metric_scores.items():
                sorted_metric_scores: Dict[str, List[float]] = dict(
                    sorted(list(values.items()), key=lambda item: item[0])
                )

                # final_score may be absent in some aggregator files after filtering.
                display_scores: Dict[str, Any] = {}

                if "final_score" in sorted_metric_scores:
                    final_score_value, final_score_count = sorted_metric_scores["final_score"]
                    display_scores[f"all ({final_score_count})"] = final_score_value

                for scenario_type, score_pair in sorted_metric_scores.items():
                    if scenario_type == "final_score":
                        continue
                    display_scores[f"{scenario_type} ({score_pair[1]})"] = score_pair[0]

                if not display_scores:
                    continue

                metric_file_scenario_types = list(display_scores.keys())
                metric_file_scenario_type_scores = list(display_scores.values())
                planner_name_columns[planner_name] += metric_file_scenario_type_scores

                if not metric_aggregator_file_scenario_types:
                    metric_aggregator_file_scenario_types = metric_file_scenario_types

                metric_file_rows = max(metric_file_rows, len(metric_file_scenario_types))

            if metric_file_rows > 0:
                metric_aggregator_files += [metric_file] + [""] * (metric_file_rows - 1)
                scenario_types += metric_aggregator_file_scenario_types

            for planner_name in planner_names:
                if planner_name not in metric_scores:
                    planner_name_columns[planner_name] += ["-"] * metric_file_rows
                else:
                    current_len = len(planner_name_columns[planner_name])
                    target_len = len(metric_aggregator_files)
                    if current_len < target_len:
                        planner_name_columns[planner_name] += ["-"] * (target_len - current_len)

        if planner_name_columns:
            data_sources: Dict[str, List[Any]] = {
                "experiment": metric_aggregator_files,
                "scenario_type": scenario_types,
            }
            data_sources.update(planner_name_columns)

            planner_table_columns = [
                TableColumn(field=planner_name, title=f"Evaluation Score {index + 1}: {planner_name}", sortable=False)
                for index, planner_name in enumerate(planner_name_columns.keys())
            ]
            columns = [
                TableColumn(**OverviewTabExperimentTableColumnConfig.get_config()),
                TableColumn(**OverviewTabScenarioTypeTableColumnConfig.get_config()),
            ]
            columns += planner_table_columns
            self.table.columns = columns
            self.table.source.data = data_sources
        else:
            self.table.columns = self._default_columns
            self.table.source.data = self._default_datasource_dict
