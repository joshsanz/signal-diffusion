"""HPO utilities for result loading, parsing, and formatting."""

from .config import (
    SPEC_TYPES,
    TASK_TYPES,
    SPEC_TO_CONFIG,
    TASK_TYPE_TO_TASKS,
    get_base_config_path,
    get_tasks_for_type,
    get_optimize_task_arg,
)
from .formatting import HPOResultsFormatter
from .metrics import (
    TaskMetrics,
    TrialSummary,
    extract_task_metrics,
    parse_best_trial,
)
from .results import (
    HPOStudyResult,
    extract_spec_type_and_task,
    find_hpo_results,
    load_hpo_result,
    parse_hpo_result,
)

__all__ = [
    # Config
    "SPEC_TYPES",
    "TASK_TYPES",
    "SPEC_TO_CONFIG",
    "TASK_TYPE_TO_TASKS",
    "get_base_config_path",
    "get_tasks_for_type",
    "get_optimize_task_arg",
    # Formatting
    "HPOResultsFormatter",
    # Metrics
    "TaskMetrics",
    "TrialSummary",
    "extract_task_metrics",
    "parse_best_trial",
    # Results
    "HPOStudyResult",
    "extract_spec_type_and_task",
    "find_hpo_results",
    "load_hpo_result",
    "parse_hpo_result",
]
