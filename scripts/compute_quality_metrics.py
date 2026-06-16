#!/usr/bin/env python3
"""Compute OmniCSEval quality metrics by scenario and overall.

Reads benchmark/OmniCSEval.json and main-evaluation/* only. It prints one
table per evaluation directory: 28 model folders plus Baseline-Human-Reference.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_REL = Path("benchmark") / "OmniCSEval.json"
MAIN_EVALUATION_REL = Path("main-evaluation")
HUMAN_REFERENCE_DIR = "Baseline-Human-Reference"
EXPECTED_SAMPLE_COUNT = 1800
EXPECTED_MODEL_COUNT = 29
SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
ALLOWED_CONCLUSIONS = {SUPPORTED, NOT_SUPPORTED}
DISPLAY_SCENARIOS = [
    ("Screenplay", "Screenplay"),
    ("Media-Interview", "Interview"),
    ("Daily-Life", "Daily"),
    ("Meeting", "Meeting"),
    ("Healthcare", "Healthcare"),
    ("Customer-Service", "Customer"),
]
RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"
RESET = "\033[0m"


@dataclass(frozen=True)
class SampleMetrics:
    completeness: float
    conciseness: float
    faithfulness: float
    reasoning_tokens: int | None


@dataclass(frozen=True)
class AggregateMetrics:
    completeness: float
    conciseness: float
    faithfulness: float
    reasoning_tokens: float | None

    def as_tuple(self) -> tuple[float, float, float, float | None]:
        return self.completeness, self.conciseness, self.faithfulness, self.reasoning_tokens


class DataError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / BENCHMARK_REL
    data = load_json(path)
    if not isinstance(data, list):
        raise DataError(f"{path} must contain a JSON list")
    if len(data) != EXPECTED_SAMPLE_COUNT:
        raise DataError(f"{path} must contain {EXPECTED_SAMPLE_COUNT} samples, got {len(data)}")
    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise DataError(f"{path}[{index}] must be a JSON object")
        if not isinstance(sample.get("scenario"), str) or not sample["scenario"]:
            raise DataError(f"{path}[{index}].scenario must be a non-empty string")
    return data


def scenario_indices(benchmark: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], dict[str, list[int]]]:
    found_scenarios: list[str] = []
    by_scenario: dict[str, list[int]] = {}
    for index, sample in enumerate(benchmark):
        scenario = sample["scenario"]
        if scenario not in by_scenario:
            found_scenarios.append(scenario)
            by_scenario[scenario] = []
        by_scenario[scenario].append(index)

    if len(found_scenarios) != 6:
        raise DataError(f"Expected 6 scenarios, found {len(found_scenarios)}: {found_scenarios}")
    expected_sources = [source for source, _label in DISPLAY_SCENARIOS]
    missing = sorted(set(expected_sources) - set(by_scenario))
    extra = sorted(set(by_scenario) - set(expected_sources))
    if missing or extra:
        raise DataError(f"Scenario mismatch: missing={missing}, extra={extra}")
    bad_counts = {name: len(indices) for name, indices in by_scenario.items() if len(indices) != 300}
    if bad_counts:
        raise DataError(f"Expected 300 samples per scenario, got {bad_counts}")
    return DISPLAY_SCENARIOS, by_scenario


def evaluation_dirs(project_root: Path, requested_models: list[str] | None) -> list[Path]:
    main_dir = project_root / MAIN_EVALUATION_REL
    dirs = sorted([path for path in main_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if requested_models:
        wanted = set(requested_models)
        found = {path.name for path in dirs}
        missing = sorted(wanted - found)
        if missing:
            raise DataError(f"Requested evaluation directories not found: {missing}")
        dirs = [path for path in dirs if path.name in wanted]
    else:
        if len(dirs) != EXPECTED_MODEL_COUNT:
            raise DataError(f"Expected {EXPECTED_MODEL_COUNT} evaluation directories, found {len(dirs)}")
        dirs = sorted(dirs, key=lambda p: (p.name != HUMAN_REFERENCE_DIR, p.name))
    return dirs


def require_dict(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise DataError(f"{path} must contain a JSON object")
    return data


def require_list(path: Path) -> list[Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise DataError(f"{path} must contain a JSON list")
    return data


def require_numeric_alignment(path: Path, item: dict[str, Any], position: int) -> int | None:
    value = item.get("align_sentence_idx")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(
            f"{path}[{position}].align_sentence_idx must be an integer or null, got {value!r}"
        )
    return value


def sample_paths(eval_dir: Path, index: int) -> tuple[Path, Path, Path]:
    return (
        eval_dir / f"sample{index}_bundle.json",
        eval_dir / f"sample{index}_key-facts-matching.json",
        eval_dir / f"sample{index}_summary-facts-verification.json",
    )


def compute_sample_metrics(eval_dir: Path, index: int) -> SampleMetrics:
    bundle_path, key_matching_path, verification_path = sample_paths(eval_dir, index)
    for path in (bundle_path, key_matching_path, verification_path):
        if not path.is_file():
            raise DataError(f"Missing required file: {path}")

    bundle = require_dict(bundle_path)
    key_matching = require_list(key_matching_path)
    verification = require_list(verification_path)

    summ_sents = bundle.get("summ_sents")
    if not isinstance(summ_sents, list) or not all(isinstance(item, str) for item in summ_sents):
        raise DataError(f"{bundle_path}.summ_sents must be a string list")
    if not summ_sents:
        raise DataError(f"{bundle_path}.summ_sents is empty; conciseness denominator would be 0")
    if not key_matching:
        raise DataError(f"{key_matching_path} is empty; completeness denominator would be 0")
    if not verification:
        raise DataError(f"{verification_path} is empty; faithfulness denominator would be 0")

    reasoning_tokens = bundle.get("reasoning_tokens")
    if reasoning_tokens is not None:
        if isinstance(reasoning_tokens, bool) or not isinstance(reasoning_tokens, int):
            raise DataError(f"{bundle_path}.reasoning_tokens must be an integer when present")
        if reasoning_tokens < 0:
            raise DataError(f"{bundle_path}.reasoning_tokens must not be negative")

    aligned_key_facts = 0
    aligned_sentence_indices: set[int] = set()
    for position, item in enumerate(key_matching):
        if not isinstance(item, dict):
            raise DataError(f"{key_matching_path}[{position}] must be a JSON object")
        align_idx = require_numeric_alignment(key_matching_path, item, position)
        if align_idx is None:
            continue
        if align_idx < 0 or align_idx >= len(summ_sents):
            raise DataError(
                f"{key_matching_path}[{position}].align_sentence_idx={align_idx} is out of "
                f"range for {len(summ_sents)} summary sentences"
            )
        aligned_key_facts += 1
        aligned_sentence_indices.add(align_idx)

    supported_summary_facts = 0
    for position, item in enumerate(verification):
        if not isinstance(item, dict):
            raise DataError(f"{verification_path}[{position}] must be a JSON object")
        if item.get("fact_idx") != position:
            raise DataError(
                f"{verification_path}[{position}].fact_idx must be {position}, "
                f"got {item.get('fact_idx')!r}"
            )
        conclusion = item.get("conclusion")
        if conclusion not in ALLOWED_CONCLUSIONS:
            raise DataError(
                f"{verification_path}[{position}].conclusion must be supported/not supported, "
                f"got {conclusion!r}"
            )
        if conclusion == SUPPORTED:
            supported_summary_facts += 1

    return SampleMetrics(
        completeness=aligned_key_facts / len(key_matching),
        conciseness=len(aligned_sentence_indices) / len(summ_sents),
        faithfulness=supported_summary_facts / len(verification),
        reasoning_tokens=reasoning_tokens,
    )


def aggregate(metrics: Iterable[SampleMetrics]) -> AggregateMetrics:
    items = list(metrics)
    if not items:
        raise DataError("Cannot aggregate an empty metrics list")
    token_values = [item.reasoning_tokens for item in items if item.reasoning_tokens is not None]
    if token_values and len(token_values) != len(items):
        raise DataError("reasoning_tokens is present for only part of an aggregation group")
    return AggregateMetrics(
        completeness=statistics.fmean(item.completeness for item in items),
        conciseness=statistics.fmean(item.conciseness for item in items),
        faithfulness=statistics.fmean(item.faithfulness for item in items),
        reasoning_tokens=statistics.fmean(token_values) if token_values else None,
    )


def scenario_std(scenario_aggregates: Iterable[AggregateMetrics]) -> AggregateMetrics:
    items = list(scenario_aggregates)
    if len(items) != 6:
        raise DataError(f"Expected 6 scenario aggregates for std, got {len(items)}")
    return AggregateMetrics(
        completeness=statistics.pstdev(item.completeness for item in items),
        conciseness=statistics.pstdev(item.conciseness for item in items),
        faithfulness=statistics.pstdev(item.faithfulness for item in items),
        reasoning_tokens=None,
    )


def format_percent(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value * 100:.1f}"


def format_percent_std(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value * 100:.2f}"


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return "nan"
    return f"{value:.1f}"


def format_number_std(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return "nan"
    return f"{value:.2f}"


def display_model_name(model_name: str, use_color: bool) -> str:
    if model_name == HUMAN_REFERENCE_DIR:
        label = "Human Reference"
        return f"{RED_BOLD}{label}{RESET}" if use_color else label
    return f"{GREEN_BOLD}{model_name}{RESET}" if use_color else model_name


def render_table(
    model_name: str,
    scenario_names: list[tuple[str, str]],
    scenario_aggs: dict[str, AggregateMetrics],
    overall: AggregateMetrics,
    std: AggregateMetrics,
    use_color: bool,
) -> str:
    rows = []
    show_reasoning_tokens = overall.reasoning_tokens is not None
    for scenario_source, scenario_label in scenario_names:
        agg = scenario_aggs[scenario_source]
        values = [
            scenario_label,
            format_percent(agg.completeness),
            format_percent(agg.conciseness),
            format_percent(agg.faithfulness),
        ]
        if show_reasoning_tokens:
            values.append(format_number(agg.reasoning_tokens))
        rows.append(values)
    overall_values = [
        "Overall",
        format_percent(overall.completeness),
        format_percent(overall.conciseness),
        format_percent(overall.faithfulness),
    ]
    if show_reasoning_tokens:
        overall_values.append(format_number(overall.reasoning_tokens))
    rows.append(overall_values)

    std_values = [
        "Scenario-Std",
        format_percent_std(std.completeness),
        format_percent_std(std.conciseness),
        format_percent_std(std.faithfulness),
    ]
    if show_reasoning_tokens:
        std_values.append("")
    rows.append(std_values)

    headers = ["Scenario", "Completeness", "Conciseness", "Faithfulness"]
    if show_reasoning_tokens:
        headers.append("Reasoning Tokens")
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    def line(left: str, mid: str, right: str, fill: str = "─") -> str:
        return left + mid.join(fill * (width + 2) for width in widths) + right

    def row(values: list[str]) -> str:
        return (
            "│ "
            + " │ ".join(
                value.ljust(widths[col]) if col == 0 else value.rjust(widths[col])
                for col, value in enumerate(values)
            )
            + " │"
        )

    title = display_model_name(model_name, use_color)
    output = [title, line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
    for idx, values in enumerate(rows):
        if idx == len(scenario_names) or idx == len(scenario_names) + 1:
            output.append(line("├", "┼", "┤"))
        output.append(row(values))
    output.append(line("└", "┴", "┘"))
    return "\n".join(output)


def compute_for_eval_dir(
    eval_dir: Path,
    scenario_names: list[tuple[str, str]],
    by_scenario: dict[str, list[int]],
) -> tuple[dict[str, AggregateMetrics], AggregateMetrics, AggregateMetrics]:
    all_metrics: list[SampleMetrics | None] = [None] * EXPECTED_SAMPLE_COUNT
    for index in range(EXPECTED_SAMPLE_COUNT):
        all_metrics[index] = compute_sample_metrics(eval_dir, index)

    scenario_aggs: dict[str, AggregateMetrics] = {}
    for scenario_source, _scenario_label in scenario_names:
        indices = by_scenario[scenario_source]
        scenario_aggs[scenario_source] = aggregate(
            all_metrics[index] for index in indices if all_metrics[index]
        )

    overall = aggregate(item for item in all_metrics if item)
    std = scenario_std(scenario_aggs[scenario_source] for scenario_source, _label in scenario_names)
    return scenario_aggs, overall, std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Completeness, Conciseness, and Faithfulness for all OmniCSEval "
            "evaluation directories by scenario and overall."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project root. Default: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional evaluation directory names to compute. Default: all 29 directories.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final validation summary, not per-model tables.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in model titles.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    try:
        benchmark = load_benchmark(project_root)
        scenario_names, by_scenario = scenario_indices(benchmark)
        eval_dirs = evaluation_dirs(project_root, args.models)

        for eval_dir in eval_dirs:
            scenario_aggs, overall, std = compute_for_eval_dir(
                eval_dir, scenario_names, by_scenario
            )
            if not args.quiet:
                print(
                    render_table(
                        eval_dir.name,
                        scenario_names,
                        scenario_aggs,
                        overall,
                        std,
                        use_color=not args.no_color,
                    )
                )
                print()

        scenario_labels = [label for _source, label in scenario_names]
        print(
            f"Computed {len(eval_dirs)} evaluation directories, "
            f"{EXPECTED_SAMPLE_COUNT} samples each, scenarios={scenario_labels}."
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
