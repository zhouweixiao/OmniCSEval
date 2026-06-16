#!/usr/bin/env python3
"""Compute meta-evaluation stability and leaderboard robustness.

This script reads meta-evaluation/stability_robustness and writes nothing.

Stability:
  - Summary-Level: Krippendorff's alpha (interval) across DeepSeek run1/2/3.
  - System-Level: Kendall's W across DeepSeek run1/2/3 model rankings.

Leaderboard robustness:
  - Spearman correlation over 28 model-level scores from three judges:
    DeepSeek run1, GLM, and Kimi.
  - Model-level scores are rounded to one decimal percentage before ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
STABILITY_REL = Path("meta-evaluation") / "stability_robustness"
EXPECTED_MODEL_COUNT = 28
EXPECTED_SAMPLES_PER_MODEL = 300
SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
ALLOWED_CONCLUSIONS = {SUPPORTED, NOT_SUPPORTED}

DIMENSIONS = ["Completeness", "Conciseness", "Faithfulness"]
DEEPSEEK_RUNS = [
    "deepseek-v3.2-instruct-run1",
    "deepseek-v3.2-instruct-run2",
    "deepseek-v3.2-instruct-run3",
]
ROBUSTNESS_JUDGES = [
    ("DeepSeek-V3.2-Instruct", "deepseek-v3.2-instruct-run1"),
    ("GLM-4.7-Instruct", "glm-4.7-instruct"),
    ("Kimi-K2-Instruct", "kimi-k2-instruct"),
]
ALL_JUDGE_SUFFIXES = list(dict.fromkeys(DEEPSEEK_RUNS + [suffix for _name, suffix in ROBUSTNESS_JUDGES]))

BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
RESET = "\033[0m"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampleMetrics:
    completeness: float
    conciseness: float
    faithfulness: float

    def value(self, dimension: str) -> float:
        if dimension == "Completeness":
            return self.completeness
        if dimension == "Conciseness":
            return self.conciseness
        if dimension == "Faithfulness":
            return self.faithfulness
        raise KeyError(dimension)


def color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{RESET}" if use_color else text


class ProgressBar:
    def __init__(self, total: int, enabled: bool, label: str = "Loading metrics") -> None:
        self.total = total
        self.enabled = enabled
        self.label = label
        self.current = 0
        self.last_rendered = -1
        self.render_step = max(1, total // 200)

    def update(self, model_name: str = "") -> None:
        if not self.enabled:
            return
        self.current += 1
        if self.current != self.total and self.current - self.last_rendered < self.render_step:
            return

        ratio = self.current / self.total if self.total else 1.0
        width = 32
        filled = min(width, int(ratio * width))
        bar = "#" * filled + "-" * (width - filled)
        suffix = f"  {model_name}" if model_name else ""
        sys.stderr.write(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} "
            f"({ratio * 100:5.1f}%){suffix}"
        )
        sys.stderr.flush()
        self.last_rendered = self.current

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_dirs(stability_dir: Path) -> list[Path]:
    dirs = sorted([path for path in stability_dir.iterdir() if path.is_dir()], key=lambda p: p.name)
    if len(dirs) != EXPECTED_MODEL_COUNT:
        raise DataError(f"Expected {EXPECTED_MODEL_COUNT} model dirs, got {len(dirs)}")
    return dirs


def sample_indices(model_dir: Path) -> list[int]:
    indices: list[int] = []
    for path in model_dir.glob("sample*_summary-sentences.json"):
        index_text = path.name.removeprefix("sample").removesuffix("_summary-sentences.json")
        if not index_text.isdigit():
            raise DataError(f"Unexpected summary sentence filename: {path}")
        indices.append(int(index_text))
    indices = sorted(indices)
    if len(indices) != EXPECTED_SAMPLES_PER_MODEL:
        raise DataError(
            f"Expected {EXPECTED_SAMPLES_PER_MODEL} samples in {model_dir.name}, got {len(indices)}"
        )
    return indices


def require_string_list_dict(path: Path, key: str) -> list[str]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise DataError(f"{path} must contain a JSON object")
    if list(data.keys()) != [key]:
        raise DataError(f"{path} keys must be exactly [{key!r}]")
    values = data[key]
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise DataError(f"{path}.{key} must be a string list")
    return values


def require_list(path: Path) -> list[Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise DataError(f"{path} must contain a JSON list")
    return data


def compute_sample_metrics(model_dir: Path, index: int, judge_suffix: str) -> SampleMetrics:
    summary_sentences_path = model_dir / f"sample{index}_summary-sentences.json"
    summary_facts_path = model_dir / f"sample{index}_summary-facts.json"
    key_matching_path = model_dir / f"sample{index}_key-facts-matching_{judge_suffix}.json"
    verification_path = model_dir / f"sample{index}_summary-facts-verification_{judge_suffix}.json"

    for path in [summary_sentences_path, summary_facts_path, key_matching_path, verification_path]:
        if not path.is_file():
            raise DataError(f"Missing required file: {path}")

    summ_sents = require_string_list_dict(summary_sentences_path, "summ_sents")
    atomic_facts = require_string_list_dict(summary_facts_path, "atomic_facts")
    key_matching = require_list(key_matching_path)
    verification = require_list(verification_path)

    if not summ_sents:
        raise DataError(f"{summary_sentences_path}.summ_sents is empty")
    if not key_matching:
        raise DataError(f"{key_matching_path} is empty")
    if not verification:
        raise DataError(f"{verification_path} is empty")
    if len(verification) != len(atomic_facts):
        raise DataError(
            f"{verification_path} length {len(verification)} does not match "
            f"{summary_facts_path}.atomic_facts length {len(atomic_facts)}"
        )

    aligned_key_facts = 0
    aligned_sentence_indices: set[int] = set()
    for pos, item in enumerate(key_matching):
        if not isinstance(item, dict):
            raise DataError(f"{key_matching_path}[{pos}] must be a JSON object")
        align_idx = item.get("align_sentence_idx")
        if align_idx is None:
            continue
        if isinstance(align_idx, bool) or not isinstance(align_idx, int):
            raise DataError(f"{key_matching_path}[{pos}].align_sentence_idx must be int or null")
        if align_idx < 0 or align_idx >= len(summ_sents):
            raise DataError(
                f"{key_matching_path}[{pos}].align_sentence_idx={align_idx} out of range "
                f"for {len(summ_sents)} summary sentences"
            )
        aligned_key_facts += 1
        aligned_sentence_indices.add(align_idx)

    supported_facts = 0
    for pos, item in enumerate(verification):
        if not isinstance(item, dict):
            raise DataError(f"{verification_path}[{pos}] must be a JSON object")
        if set(item) != {"fact_idx", "conclusion", "reason"}:
            raise DataError(f"{verification_path}[{pos}] keys must be fact_idx/conclusion/reason")
        if item["fact_idx"] != pos:
            raise DataError(
                f"{verification_path}[{pos}].fact_idx must be {pos}, got {item['fact_idx']!r}"
            )
        if item["conclusion"] not in ALLOWED_CONCLUSIONS:
            raise DataError(
                f"{verification_path}[{pos}].conclusion must be supported/not supported"
            )
        if not isinstance(item["reason"], str):
            raise DataError(f"{verification_path}[{pos}].reason must be a string")
        if item["conclusion"] == SUPPORTED:
            supported_facts += 1

    return SampleMetrics(
        completeness=aligned_key_facts / len(key_matching),
        conciseness=len(aligned_sentence_indices) / len(summ_sents),
        faithfulness=supported_facts / len(verification),
    )


def krippendorff_alpha_interval(ratings_by_unit: list[list[float]]) -> float:
    cleaned = [unit for unit in ratings_by_unit if len(unit) >= 2]
    all_values = [value for unit in cleaned for value in unit]
    n_total = len(all_values)
    if n_total < 2:
        return math.nan

    observed_numerator = 0.0
    for unit in cleaned:
        unit_sum = 0.0
        for a in unit:
            for b in unit:
                unit_sum += (a - b) ** 2
        observed_numerator += unit_sum / (len(unit) - 1)
    observed_disagreement = observed_numerator / n_total

    grand_mean = statistics.fmean(all_values)
    total_sum_squares = sum((value - grand_mean) ** 2 for value in all_values)
    expected_disagreement = 2.0 * total_sum_squares / (n_total - 1)
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else math.nan
    return 1.0 - observed_disagreement / expected_disagreement


def average_ranks_desc(values: list[float]) -> tuple[list[float], float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    ranks = [0.0] * len(values)
    tie_correction = 0.0
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        tie_size = end - pos
        if tie_size > 1:
            tie_correction += tie_size**3 - tie_size
        for idx in range(pos, end):
            ranks[indexed[idx][0]] = avg_rank
        pos = end
    return ranks, tie_correction


def kendalls_w_tie_corrected(score_rows: list[list[float]]) -> float:
    if not score_rows:
        return math.nan
    m = len(score_rows)
    n = len(score_rows[0])
    if n < 2 or any(len(row) != n for row in score_rows):
        raise DataError("Kendall's W score rows must have equal length >= 2")

    rank_rows: list[list[float]] = []
    tie_correction_total = 0.0
    for row in score_rows:
        ranks, tie_correction = average_ranks_desc(row)
        rank_rows.append(ranks)
        tie_correction_total += tie_correction

    rank_sums = [sum(rank_rows[rater][obj] for rater in range(m)) for obj in range(n)]
    mean_rank_sum = m * (n + 1) / 2.0
    s_value = sum((rank_sum - mean_rank_sum) ** 2 for rank_sum in rank_sums)
    denominator = m**2 * (n**3 - n) - m * tie_correction_total
    if denominator == 0:
        return math.nan
    return 12.0 * s_value / denominator


def format_score(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def format_pvalue(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3e}"


def system_score(
    metrics: dict[str, dict[int, dict[str, SampleMetrics]]],
    model_name: str,
    indices: list[int],
    judge_suffix: str,
    dimension: str,
) -> float:
    values = [metrics[model_name][index][judge_suffix].value(dimension) for index in indices]
    return round(statistics.fmean(values) * 100.0, 1)


def load_all_metrics(
    stability_dir: Path,
    show_progress: bool,
) -> tuple[list[str], list[int], dict[str, dict[int, dict[str, SampleMetrics]]]]:
    models = model_dirs(stability_dir)
    model_names = [model_dir.name for model_dir in models]
    reference_indices: list[int] | None = None
    metrics: dict[str, dict[int, dict[str, SampleMetrics]]] = {}
    progress = ProgressBar(
        total=len(models) * EXPECTED_SAMPLES_PER_MODEL * len(ALL_JUDGE_SUFFIXES),
        enabled=show_progress,
    )

    try:
        for model_dir in models:
            indices = sample_indices(model_dir)
            if reference_indices is None:
                reference_indices = indices
            elif indices != reference_indices:
                raise DataError(f"Sample index set mismatch for {model_dir.name}")

            metrics[model_dir.name] = {}
            for index in indices:
                metrics[model_dir.name][index] = {}
                for suffix in ALL_JUDGE_SUFFIXES:
                    metrics[model_dir.name][index][suffix] = compute_sample_metrics(
                        model_dir, index, suffix
                    )
                    progress.update(model_dir.name)
    finally:
        progress.finish()

    return model_names, reference_indices or [], metrics


def compute_stability(
    model_names: list[str],
    indices: list[int],
    metrics: dict[str, dict[int, dict[str, SampleMetrics]]],
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for dimension in DIMENSIONS:
        ratings_by_unit = [
            [metrics[model_name][index][run].value(dimension) for run in DEEPSEEK_RUNS]
            for model_name in model_names
            for index in indices
        ]
        summary_level = krippendorff_alpha_interval(ratings_by_unit)

        run_score_rows = [
            [system_score(metrics, model_name, indices, run, dimension) for model_name in model_names]
            for run in DEEPSEEK_RUNS
        ]
        system_level = kendalls_w_tie_corrected(run_score_rows)
        results[dimension] = (system_level, summary_level)
    return results


def compute_robustness(
    model_names: list[str],
    indices: list[int],
    metrics: dict[str, dict[int, dict[str, SampleMetrics]]],
) -> dict[str, list[list[tuple[float, float]]]]:
    judge_labels = [name for name, _suffix in ROBUSTNESS_JUDGES]
    scores = {
        judge_name: {
            dimension: {
                model_name: system_score(metrics, model_name, indices, judge_suffix, dimension)
                for model_name in model_names
            }
            for dimension in DIMENSIONS
        }
        for judge_name, judge_suffix in ROBUSTNESS_JUDGES
    }

    results: dict[str, list[list[tuple[float, float]]]] = {}
    for dimension in DIMENSIONS:
        matrix: list[list[tuple[float, float]]] = []
        for row_judge in judge_labels:
            row: list[tuple[float, float]] = []
            row_scores = [scores[row_judge][dimension][model_name] for model_name in model_names]
            for col_judge in judge_labels:
                col_scores = [scores[col_judge][dimension][model_name] for model_name in model_names]
                result = spearmanr(row_scores, col_scores)
                row.append((float(result.statistic), float(result.pvalue)))
            matrix.append(row)
        results[dimension] = matrix
    return results


def visible_len(text: str) -> int:
    return len(ANSI_RE.sub("", text))


def pad_visible(text: str, width: int, align: str = "left") -> str:
    padding = max(0, width - visible_len(text))
    if align == "right":
        return " " * padding + text
    return text + " " * padding


def render_box_table(
    title: str | None,
    headers: list[str],
    rows: list[list[str]],
    use_color: bool,
    numeric_columns: set[int] | None = None,
) -> str:
    numeric_columns = numeric_columns or set()
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(values: list[str], is_header: bool = False) -> str:
        rendered = []
        for col, value in enumerate(values):
            align = "right" if col in numeric_columns and not is_header else "left"
            padded = pad_visible(value, widths[col], align)
            if is_header and value:
                padded = color(padded, CYAN, use_color)
            elif col == 0 and value:
                padded = color(padded, CYAN, use_color)
            rendered.append(padded)
        return "| " + " | ".join(rendered) + " |"

    lines: list[str] = []
    if title:
        lines.append(color(title, BLUE, use_color))
    lines.extend([border(), render_row(headers, is_header=True), border()])
    lines.extend(render_row(row) for row in rows)
    lines.append(border())
    return "\n".join(lines)


def render_stability(results: dict[str, tuple[float, float]], use_color: bool) -> str:
    rows = [
        [dimension, format_score(system_level), format_score(summary_level)]
        for dimension, (system_level, summary_level) in results.items()
    ]
    return render_box_table(
        "DeepSeek-V3.2-Instruct Stability",
        ["Dimension", "System-Level", "Summary-Level"],
        rows,
        use_color,
        numeric_columns={1, 2},
    )


def render_robustness(
    results: dict[str, list[list[tuple[float, float]]]], use_color: bool
) -> str:
    judge_labels = [name for name, _suffix in ROBUSTNESS_JUDGES]
    rendered_sections: list[str] = []
    for dimension in DIMENSIONS:
        rows: list[list[str]] = []
        matrix = results[dimension]
        for judge_label, values in zip(judge_labels, matrix):
            rows.append(
                [judge_label]
                + [
                    f"{format_score(rho)} (p={format_pvalue(p_value)})"
                    for rho, p_value in values
                ]
            )
        rendered_sections.append(
            render_box_table(
                None,
                [dimension] + judge_labels,
                rows,
                use_color,
            )
        )
    return "\n\n".join(rendered_sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute OmniCSEval meta-evaluation stability and leaderboard robustness."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project root. Default: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the stderr progress bar while loading sample metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    stability_dir = project_root / STABILITY_REL

    try:
        model_names, indices, metrics = load_all_metrics(
            stability_dir,
            show_progress=not args.no_progress,
        )
        stability_results = compute_stability(model_names, indices, metrics)
        robustness_results = compute_robustness(model_names, indices, metrics)

        use_color = not args.no_color
        print(render_stability(stability_results, use_color))
        print()
        print(color("Leaderboard Robustness", GREEN, use_color))
        print(render_robustness(robustness_results, use_color))
        print()
        print(
            f"Checked {len(model_names)} models x {len(indices)} samples "
            f"= {len(model_names) * len(indices)} summaries."
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
