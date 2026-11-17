#!/usr/bin/env python3
"""Compute PURE task, capability, and macro metrics from result JSONL."""

import argparse
import json
import os
import re
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "questions.jsonl"
TASK_ORDER = (
    "A1", "A2", "A3", "B1", "B2", "B3", "B4",
    "B5", "C1", "C2", "C3", "C4", "D1", "D2",
)
DIMENSIONS = {
    "perception": {
        "label": "Perception",
        "tasks": ("A1", "A2", "A3"),
    },
    "understanding": {
        "label": "Understanding",
        "tasks": ("B1", "B2", "B3", "B4", "B5"),
    },
    "reasoning": {
        "label": "Reasoning",
        "tasks": ("C1", "C2", "C3", "C4"),
    },
    "judgment": {
        "label": "Judgment",
        "tasks": ("D1", "D2"),
    },
}
ANSWER_PARSING_POLICY = "answer_field_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSONL {} line {}".format(path, line_number)) from exc
    return rows


def index_unique(rows, label):
    indexed = {}
    for row in rows:
        question_id = row.get("question_id")
        if not question_id or question_id in indexed:
            raise ValueError("{} has missing or duplicate question_id: {!r}".format(label, question_id))
        indexed[question_id] = row
    return indexed


def normalize_answer(value, valid_options):
    if not isinstance(value, str):
        return None
    answer = value.strip().upper()
    if (
        not answer
        or len(answer) != len(set(answer))
        or any(character not in valid_options for character in answer)
    ):
        return None
    return "".join(sorted(answer))


def parse_json_answer(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = raw.strip()
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            return payload["answer"]
    except json.JSONDecodeError:
        pass
    matches = re.findall(r'"answer"\s*:\s*"([^"]*)"', candidate)
    return matches[0] if len(matches) == 1 else None


def extract_answer(raw, options, task_id):
    answer = normalize_answer(parse_json_answer(raw), set(options))
    if answer is None:
        return None
    if task_id not in ("A2", "A3") and len(answer) != 1:
        return None
    return answer


def percentage(numerator, denominator):
    return round(numerator / denominator * 100, 6) if denominator else None


def mean(values):
    return round(sum(values) / len(values), 6) if values else None


def group_c3_questions(rows):
    groups = {}
    expected_sub_ids = {"img0_q0", "img0_q1", "img1_q0", "img1_q1"}
    for row in rows:
        if row["task_id"] != "C3":
            continue
        group_id, sub_id = row.get("group_id"), row.get("sub_id")
        if not isinstance(group_id, str) or not group_id or sub_id not in expected_sub_ids:
            raise ValueError("Invalid C3 group or sub-question identifier")
        group = groups.setdefault(group_id, {})
        if sub_id in group:
            raise ValueError("Duplicate C3 sub-question in {}".format(group_id))
        group[sub_id] = row["question_id"]
    for group_id, group in groups.items():
        if set(group) != expected_sub_ids:
            raise ValueError("Incomplete C3 group: {}".format(group_id))
    return {group_id: list(group.values()) for group_id, group in groups.items()}


def score_results(input_path, results_path):
    input_path = input_path.expanduser().resolve()
    results_path = results_path.expanduser().resolve()
    input_rows = read_jsonl(input_path)
    result_rows = read_jsonl(results_path)
    inputs = index_unique(input_rows, "input")
    results = index_unique(result_rows, "results")
    extra_ids = sorted(set(results) - set(inputs))
    if extra_ids:
        raise ValueError("Results contain unknown question IDs: {}".format(extra_ids[:5]))
    input_tasks = {row.get("task_id") for row in input_rows}
    if not input_rows:
        raise ValueError("Input contains no questions")
    unknown_tasks = sorted(input_tasks - set(TASK_ORDER))
    if unknown_tasks:
        raise ValueError("Input contains unknown tasks: {}".format(unknown_tasks))

    model_values = {row.get("model") for row in result_rows}
    if len(model_values) > 1:
        raise ValueError("Results mix models")

    correctness = {}
    parsed_answers = {}
    format_ok = {}
    for question_id, result in results.items():
        expected = inputs[question_id]
        for field in ("task_id", "group_id", "sub_id"):
            if result.get(field) != expected.get(field):
                raise ValueError("{} has mismatched {}".format(question_id, field))
        valid_options = set(expected["options"])
        expected_answer = normalize_answer(expected["answer"], valid_options)
        result_ground_truth = normalize_answer(result.get("ground_truth"), valid_options)
        if result_ground_truth != expected_answer:
            raise ValueError("{} has mismatched ground truth".format(question_id))
        parsed = extract_answer(result.get("model_answer"), expected["options"], expected["task_id"])
        parsed_answers[question_id] = parsed
        format_ok[question_id] = parsed is not None
        correctness[question_id] = parsed is not None and parsed == expected_answer

    expected_by_task = {task: [] for task in TASK_ORDER}
    completed_by_task = {task: [] for task in TASK_ORDER}
    for question_id, row in inputs.items():
        expected_by_task[row["task_id"]].append(question_id)
    for question_id, row in results.items():
        completed_by_task[row["task_id"]].append(question_id)

    c3_groups = group_c3_questions(input_rows)

    tasks = {}
    for task in TASK_ORDER:
        expected_ids = expected_by_task[task]
        completed_ids = completed_by_task[task]
        correct = sum(correctness[question_id] for question_id in completed_ids)
        task_complete = len(completed_ids) == len(expected_ids)
        task_metrics = {
            "status": "complete" if task_complete else "partial",
            "correct": correct,
            "completed": len(completed_ids),
            "total": len(expected_ids),
            "missing": len(expected_ids) - len(completed_ids),
            "accuracy_pct": percentage(correct, len(completed_ids)),
            "effective_metric": "accuracy",
            "effective_score_pct": percentage(correct, len(completed_ids)) if task_complete else None,
        }
        tasks[task] = task_metrics

    completed_groups = 0
    correct_groups = 0
    for question_ids in c3_groups.values():
        if all(question_id in results for question_id in question_ids):
            completed_groups += 1
            correct_groups += int(all(correctness[question_id] for question_id in question_ids))
    c3_complete = completed_groups == len(c3_groups)
    tasks["C3"].update({
        "question_accuracy_pct": tasks["C3"].pop("accuracy_pct"),
        "correct_groups": correct_groups,
        "completed_groups": completed_groups,
        "total_groups": len(c3_groups),
        "missing_groups": len(c3_groups) - completed_groups,
        "group_accuracy_pct": percentage(correct_groups, completed_groups),
        "effective_metric": "group_accuracy",
        "effective_score_pct": percentage(correct_groups, completed_groups) if c3_complete else None,
    })

    complete = len(results) == len(inputs)
    dimensions = {}
    for key, config in DIMENSIONS.items():
        scores = [tasks[task]["effective_score_pct"] for task in config["tasks"]]
        dimensions[key] = {
            "label": config["label"],
            "tasks": list(config["tasks"]),
            "aggregation": "equal_task_mean",
            "score_pct": mean(scores) if all(score is not None for score in scores) else None,
        }
    task_scores = [tasks[task]["effective_score_pct"] for task in TASK_ORDER]

    return {
        "benchmark": {
            "name": "PURE",
            "input": input_path.name,
        },
        "model": next(iter(model_values), None),
        "status": "complete" if complete else "partial",
        "answer_parsing": {
            "policy": ANSWER_PARSING_POLICY,
            "source_field": "model_answer",
        },
        "coverage": {
            "expected_questions": len(inputs),
            "completed_questions": len(results),
            "missing_questions": len(inputs) - len(results),
            "parse_failed": sum(answer is None for answer in parsed_answers.values()),
            "format_failed": sum(not valid for valid in format_ok.values()),
        },
        "tasks": tasks,
        "dimensions": dimensions,
        "macro_14": {
            "aggregation": "equal_task_mean",
            "task_count": len(TASK_ORDER),
            "c3_metric": "group_accuracy",
            "score_pct": mean(task_scores) if complete and all(score is not None for score in task_scores) else None,
        },
    }


def write_metrics(input_path, results_path, output_path=None):
    results_path = results_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve() if output_path else results_path.parent / "metrics.json"
    protected = {input_path.expanduser().resolve(), results_path}
    if output_path.resolve() in protected or (output_path.exists() and any(output_path.samefile(path) for path in protected)):
        raise ValueError("Metrics output must not overwrite questions or results")
    metrics = score_results(input_path, results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    return metrics, output_path


def format_percentage(value):
    return "N/A" if value is None else "{:.2f}%".format(value)


def print_metrics(metrics):
    print("\nTask metrics")
    for task in TASK_ORDER:
        row = metrics["tasks"][task]
        if task == "C3":
            detail = "G-Acc={} ({}/{}) Q-Acc={}".format(
                format_percentage(row["group_accuracy_pct"]),
                row["correct_groups"],
                row["completed_groups"],
                format_percentage(row["question_accuracy_pct"]),
            )
        else:
            detail = "Acc={} ({}/{})".format(
                format_percentage(row["accuracy_pct"]),
                row["correct"],
                row["completed"],
            )
        print("  {:<3} {}".format(task, detail))
    print("\nCapability metrics")
    for row in metrics["dimensions"].values():
        print("  {}: {}".format(row["label"], format_percentage(row["score_pct"])))
    print("\nMacro-14: {}".format(format_percentage(metrics["macro_14"]["score_pct"])))
    # One-line per-task summary for quick copy-paste
    model_name = metrics.get("model", "?")
    cells = []
    for task in TASK_ORDER:
        v = metrics["tasks"][task]["effective_score_pct"]
        cells.append("{:.2f}".format(v) if v is not None else "-")
    macro = metrics["macro_14"]["score_pct"]
    cells.append("{:.2f}".format(macro) if macro is not None else "-")
    print("\n{} | {}".format(model_name, " | ".join(cells)))


def main():
    args = parse_args()
    metrics, output_path = write_metrics(
        args.input, args.results, args.output
    )
    print_metrics(metrics)
    print("\nmetrics={}".format(output_path))


if __name__ == "__main__":
    main()
