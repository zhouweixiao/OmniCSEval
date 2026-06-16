from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIGNMENT_ROOT = PROJECT_ROOT / "meta-evaluation" / "alignment_with_humans"

SAMPLE_RE = re.compile(r"^sample(\d+)_summary-sentences\.json$")

JUDGES = (
    "gpt-4o",
    "qwen3-next-80b-a3b-instruct",
    "llama-4-maverick",
    "mistral-large-3",
    "kimi-k2-instruct",
    "glm-4.7-instruct",
    "deepseek-v3.2-instruct",
)

DISPLAY_NAMES = {
    "deepseek-v3.2-instruct": "DeepSeek-V3.2-Instruct",
    "glm-4.7-instruct": "GLM-4.7-Instruct",
    "gpt-4o": "GPT-4o (Previous SOTA Judge)",
    "kimi-k2-instruct": "Kimi-K2-Instruct",
    "llama-4-maverick": "Llama 4 Maverick",
    "mistral-large-3": "Mistral Large 3",
    "qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B-Instruct",
}

SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
HEADER_STYLE = "\033[1;96m"
RESET_STYLE = "\033[0m"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def majority_vote(values: list[Any], context: str) -> Any:
    if len(values) != 3:
        raise ValueError(f"{context}: expected 3 human annotations, got {len(values)}")
    counts = Counter(repr(value) for value in values)
    top_count = max(counts.values())
    winners = [value for value in values if counts[repr(value)] == top_count]
    if top_count < 2:
        raise ValueError(f"{context}: no majority vote in {values!r}")
    return winners[0]


def require_dict(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level object, got {type(data).__name__}")
    return data


def require_list(path: Path) -> list[Any]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected top-level list, got {type(data).__name__}")
    return data


def discover_samples(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"alignment root not found: {root}")

    samples: list[tuple[Path, str]] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sentence_path in sorted(model_dir.glob("sample*_summary-sentences.json")):
            match = SAMPLE_RE.fullmatch(sentence_path.name)
            if match:
                samples.append((model_dir, match.group(1)))
    return samples


def validate_fact_idxs(records: list[Any], expected_len: int, path: Path) -> list[dict[str, Any]]:
    if len(records) != expected_len:
        raise ValueError(f"{path}: length {len(records)} != expected {expected_len}")
    cleaned: list[dict[str, Any]] = []
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{idx}]: expected object, got {type(item).__name__}")
        if item.get("fact_idx") != idx:
            raise ValueError(f"{path}[{idx}]: fact_idx {item.get('fact_idx')!r} != {idx}")
        cleaned.append(item)
    return cleaned


def align_idx_to_sentence_vector(value: Any, sent_count: int, context: str) -> list[int]:
    vector = [0] * sent_count
    if value is None:
        return vector
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context}: align_sentence_idx must be int or null, got {value!r}")
    if value < 0 or value >= sent_count:
        raise ValueError(
            f"{context}: align_sentence_idx {value} out of range for {sent_count} sentences"
        )
    vector[value] = 1
    return vector


def conclusion_to_label(value: Any, context: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{context}: conclusion must be string, got {value!r}")
    normalized = value.strip().lower()
    if normalized == SUPPORTED:
        return 1
    if normalized == NOT_SUPPORTED:
        return 0
    raise ValueError(f"{context}: unexpected conclusion {value!r}")


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"macro_f1 length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("macro_f1 received empty labels")

    f1_scores: list[float] = []
    for label in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"balanced_accuracy length mismatch: {len(y_true)} != {len(y_pred)}")
    if not y_true:
        raise ValueError("balanced_accuracy received empty labels")

    recalls: list[float] = []
    for label in (0, 1):
        positives = sum(1 for t in y_true if t == label)
        if positives == 0:
            raise ValueError(f"balanced_accuracy cannot compute recall for absent label {label}")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        recalls.append(correct / positives)
    return sum(recalls) / len(recalls)


def format_table(rows: list[tuple[str, float, float]]) -> str:
    headers = ("LLM as a Fact-Checker", "Matching (Macro F1)", "Verification (BAcc)")
    formatted_rows = [
        (name, f"{matching:.3f}", f"{verification:.3f}") for name, matching, verification in rows
    ]
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in formatted_rows)),
        max(len(headers[1]), *(len(row[1]) for row in formatted_rows)),
        max(len(headers[2]), *(len(row[2]) for row in formatted_rows)),
    ]

    def border(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (width + 2) for width in widths) + right

    def styled(value: str, enabled: bool) -> str:
        return f"{HEADER_STYLE}{value}{RESET_STYLE}" if enabled else value

    def row(values: tuple[str, str, str], highlight: bool = False) -> str:
        first = f"{values[0]:<{widths[0]}}"
        second = f"{values[1]:>{widths[1]}}"
        third = f"{values[2]:>{widths[2]}}"
        return (
            f"│ {styled(first, highlight)} │ "
            f"{styled(second, highlight)} │ "
            f"{styled(third, highlight)} │"
        )

    lines = [
        border("┌", "┬", "┐"),
        row(headers, highlight=True),
        border("├", "┼", "┤"),
    ]
    lines.extend(row(values) for values in formatted_rows)
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)


def compute_metrics(root: Path) -> tuple[list[tuple[str, float, float]], dict[str, int]]:
    samples = discover_samples(root)
    if not samples:
        raise RuntimeError(f"No samples found under {root}")

    matching_true_by_judge = {judge: [] for judge in JUDGES}
    matching_pred_by_judge = {judge: [] for judge in JUDGES}
    verification_true_by_judge = {judge: [] for judge in JUDGES}
    verification_pred_by_judge = {judge: [] for judge in JUDGES}

    stats = {
        "models": len({model_dir.name for model_dir, _sample_index in samples}),
        "samples": len(samples),
        "matching_binary_labels_per_judge": 0,
        "verification_facts_per_judge": 0,
    }

    errors: list[str] = []
    for model_dir, sample_index in samples:
        sample_context = f"{model_dir.name}/sample{sample_index}"
        try:
            sentence_path = model_dir / f"sample{sample_index}_summary-sentences.json"
            sentence_data = require_dict(sentence_path)
            if set(sentence_data.keys()) != {"summ_sents"}:
                raise ValueError(f"{sentence_path}: unexpected keys {sorted(sentence_data.keys())}")
            summ_sents = sentence_data["summ_sents"]
            if not isinstance(summ_sents, list) or not all(isinstance(s, str) for s in summ_sents):
                raise ValueError(f"{sentence_path}: summ_sents must be a string list")
            sent_count = len(summ_sents)
            if sent_count == 0:
                raise ValueError(f"{sentence_path}: summ_sents is empty")

            human_matching_path = (
                model_dir / f"sample{sample_index}_key-facts-matching_human-annotations.json"
            )
            human_matching_raw = require_list(human_matching_path)
            human_matching = validate_fact_idxs(
                human_matching_raw,
                len(human_matching_raw),
                human_matching_path,
            )

            human_matching_vectors: list[list[int]] = []
            for item in human_matching:
                values = item.get("align_sentence_idx")
                if not isinstance(values, list):
                    raise ValueError(
                        f"{human_matching_path}[{item['fact_idx']}]: align_sentence_idx must be list"
                    )
                vote = majority_vote(
                    values, f"{sample_context}/key_fact_{item['fact_idx']}/human_matching"
                )
                human_matching_vectors.append(
                    align_idx_to_sentence_vector(
                        vote,
                        sent_count,
                        f"{sample_context}/key_fact_{item['fact_idx']}/human_matching",
                    )
                )

            facts_path = model_dir / f"sample{sample_index}_summary-facts.json"
            facts_data = require_dict(facts_path)
            if set(facts_data.keys()) != {"atomic_facts"}:
                raise ValueError(f"{facts_path}: unexpected keys {sorted(facts_data.keys())}")
            atomic_facts = facts_data["atomic_facts"]
            if not isinstance(atomic_facts, list) or not all(isinstance(f, str) for f in atomic_facts):
                raise ValueError(f"{facts_path}: atomic_facts must be a string list")

            human_verification_path = (
                model_dir / f"sample{sample_index}_summary-facts-verification_human-annotations.json"
            )
            human_verification = validate_fact_idxs(
                require_list(human_verification_path), len(atomic_facts), human_verification_path
            )
            human_verification_labels = [
                conclusion_to_label(
                    majority_vote(
                        item.get("conclusion"),
                        f"{sample_context}/summary_fact_{item['fact_idx']}/human_verification",
                    ),
                    f"{sample_context}/summary_fact_{item['fact_idx']}/human_verification",
                )
                for item in human_verification
            ]

            for judge in JUDGES:
                judge_matching_path = model_dir / f"sample{sample_index}_key-facts-matching_{judge}.json"
                judge_matching = validate_fact_idxs(
                    require_list(judge_matching_path), len(human_matching), judge_matching_path
                )
                for item, true_vector in zip(judge_matching, human_matching_vectors):
                    pred_vector = align_idx_to_sentence_vector(
                        item.get("align_sentence_idx"),
                        sent_count,
                        f"{sample_context}/key_fact_{item['fact_idx']}/{judge}",
                    )
                    matching_true_by_judge[judge].extend(true_vector)
                    matching_pred_by_judge[judge].extend(pred_vector)

                judge_verification_path = (
                    model_dir / f"sample{sample_index}_summary-facts-verification_{judge}.json"
                )
                judge_verification = validate_fact_idxs(
                    require_list(judge_verification_path), len(atomic_facts), judge_verification_path
                )
                for item, true_label in zip(judge_verification, human_verification_labels):
                    pred_label = conclusion_to_label(
                        item.get("conclusion"),
                        f"{sample_context}/summary_fact_{item['fact_idx']}/{judge}",
                    )
                    verification_true_by_judge[judge].append(true_label)
                    verification_pred_by_judge[judge].append(pred_label)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sample_context}: {exc}")

    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:50]))

    rows: list[tuple[str, float, float]] = []
    for judge in JUDGES:
        matching_true = matching_true_by_judge[judge]
        matching_pred = matching_pred_by_judge[judge]
        verification_true = verification_true_by_judge[judge]
        verification_pred = verification_pred_by_judge[judge]

        if stats["matching_binary_labels_per_judge"] == 0:
            stats["matching_binary_labels_per_judge"] = len(matching_true)
        elif stats["matching_binary_labels_per_judge"] != len(matching_true):
            raise RuntimeError(f"{judge}: matching label length differs across judges")

        if stats["verification_facts_per_judge"] == 0:
            stats["verification_facts_per_judge"] = len(verification_true)
        elif stats["verification_facts_per_judge"] != len(verification_true):
            raise RuntimeError(f"{judge}: verification label length differs across judges")

        rows.append(
            (
                DISPLAY_NAMES[judge],
                macro_f1(matching_true, matching_pred),
                balanced_accuracy(verification_true, verification_pred),
            )
        )

    if not math.isfinite(sum(value for _name, value, _bacc in rows)):
        raise RuntimeError("Non-finite matching score encountered")
    if not math.isfinite(sum(value for _name, _f1, value in rows)):
        raise RuntimeError("Non-finite verification score encountered")

    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute LLM judge alignment with human annotations for fact checking."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    args = parser.parse_args()

    try:
        rows, stats = compute_metrics(args.root)
        print(format_table(rows))
        print()
        print(f"Checked {stats['models']} models, {stats['samples']} sampled summaries.")
        print(f"Matching binary labels per judge: {stats['matching_binary_labels_per_judge']}")
        print(f"Verification facts per judge: {stats['verification_facts_per_judge']}")
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
