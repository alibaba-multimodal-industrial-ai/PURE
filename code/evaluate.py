#!/usr/bin/env python3
"""Evaluate PURE through an OpenAI-compatible chat completions API."""

import argparse
import base64
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

from openai import APIStatusError, OpenAI
from tqdm import tqdm

from score_pure_results import ANSWER_PARSING_POLICY, extract_answer, group_c3_questions, normalize_answer, print_metrics, write_metrics

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "questions.jsonl"
SYSTEM_PROMPT = (
    "你是一个电商商品视觉理解专家。请根据提供的信息和问题，从给定选项中选择最准确的答案。\n"
    "如果是单选题，只选一个选项字母（如 A）；如果是多选题，必须选出所有正确选项的字母（如 ABC）。\n"
    "你必须严格按照以下 JSON 格式输出，不要任何解释、不要 markdown 代码块：\n"
    '{"answer": "在此填写最终答案的选项字母，例如 A 或 ABC"}'
)
TASK_CONFIG = {
    "A1": {"images": 1, "titles": 0, "sku_attributes": 0},
    "A2": {"images": 1, "titles": 0, "sku_attributes": 0},
    "A3": {"images": 1, "titles": 1, "sku_attributes": 0},
    "B1": {"images": 0, "titles": 0, "sku_attributes": 0},
    "B2": {"images": 1, "titles": 1, "sku_attributes": 0},
    "B3": {"images": 0, "titles": 1, "sku_attributes": 0},
    "B4": {"images": 0, "titles": 1, "sku_attributes": 1},
    "B5": {"images": 0, "titles": 0, "sku_attributes": 0},
    "C1": {"images": 2, "titles": 0, "sku_attributes": 0},
    "C2": {"images": 2, "titles": 2, "sku_attributes": 0},
    "C3": {"images": 1, "titles": 1, "sku_attributes": 0},
    "C4": {"images": 2, "titles": 0, "sku_attributes": 0},
    "D1": {"images": 2, "titles": 2, "sku_attributes": 0},
    "D2": {"images": 2, "titles": 2, "sku_attributes": 2},
}
IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
SMOKE_IDS = ("A2:3", "B2:3", "B4:2", "C1:2")


def parse_args():
    parser = argparse.ArgumentParser(description="PURE main-benchmark API evaluation")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--api-base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model identifier accepted by your provider")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None, help="Omitted from requests unless supplied")
    parser.add_argument("--max-tokens", type=int, default=None, help="Omitted from requests unless supplied")
    parser.add_argument("--smoke", action="store_true", help="Run four representative questions")
    parser.add_argument("--resume", action="store_true", help="Skip question IDs already saved in the output")
    args = parser.parse_args()
    if args.concurrency < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("concurrency and max-tokens must be positive")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature < 0):
        parser.error("temperature must be finite and nonnegative")
    target = urlsplit(args.api_base)
    if target.username or target.password or target.query or target.fragment or not target.hostname:
        parser.error("api-base must not contain credentials, a query, or a fragment")
    if target.scheme != "https" and not (target.scheme == "http" and target.hostname in {"127.0.0.1", "localhost", "::1"}):
        parser.error("api-base must use HTTPS; HTTP is allowed only for loopback")
    return args


def resolve_image_path(bundle_root, relative_path):
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Image path must be relative")
    resolved = (bundle_root / relative).resolve()
    if not resolved.is_relative_to(bundle_root) or not resolved.is_file():
        raise ValueError("Image is missing or outside the dataset directory")
    return resolved


def load_questions(input_path):
    root = input_path.parent
    rows, seen = [], set()
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            qid, task = item["question_id"], item["task_id"]
            if not qid or qid in seen or task not in TASK_CONFIG:
                raise ValueError("Invalid or duplicate question identifier or task")
            config = TASK_CONFIG[task]
            for field, count in config.items():
                if not count:
                    continue
                values = item.get(field)
                if not isinstance(values, list) or len(values) < count:
                    raise ValueError("{} requires {} entries in {}".format(qid, count, field))
                for value in values[:count]:
                    if field == "images":
                        if Path(value["path"]).suffix.lower() not in IMAGE_MEDIA_TYPES:
                            raise ValueError("Unsupported image file type")
                        resolve_image_path(root, value["path"])
                    elif not isinstance(value, str):
                        raise ValueError("{} must contain text".format(field))
            if not item["question"].strip() or not item["options"] or normalize_answer(item["answer"], set(item["options"])) is None:
                raise ValueError("Question or reference answer is invalid")
            seen.add(qid)
            rows.append(item)
    if not rows:
        raise ValueError("Dataset is empty")
    group_c3_questions(rows)
    return rows


def build_messages(item, bundle_root):
    config = TASK_CONFIG[item["task_id"]]
    evidence = {field: item[field][:count] if count else [] for field, count in config.items()}
    images, titles, sku_attributes = evidence["images"], evidence["titles"], evidence["sku_attributes"]
    content = []
    for index, image in enumerate(images, 1):
        if len(images) == 2:
            content.append({"type": "text", "text": "商品{}图：".format(index)})
        encoded = base64.b64encode(resolve_image_path(bundle_root, image["path"]).read_bytes()).decode("ascii")
        media_type = IMAGE_MEDIA_TYPES[Path(image["path"]).suffix.lower()]
        content.append({"type": "image_url", "image_url": {"url": "data:{};base64,{}".format(media_type, encoded)}})
    question_first = bool(sku_attributes) and not images
    text_parts = [item["question"]] if question_first else []
    if sku_attributes and len(images) == 2:
        text_parts.append("第1张图为商品1的主图，第2张图为商品2的主图。")
    record_count = max(len(titles), len(sku_attributes))
    for index in range(record_count):
        product = "商品{}".format(index + 1) if record_count > 1 else "商品"
        parts = []
        if index < len(titles):
            parts.append("{}标题：{}".format(product, titles[index]))
        if index < len(sku_attributes):
            prefix = product if record_count > 1 else ""
            parts.append("{}SKU属性：{}".format(prefix, sku_attributes[index]))
        text_parts.append("\n".join(parts))
    if not question_first:
        text_parts.append(item["question"])
    text_parts.append("\n选项：\n" + "\n".join("{}. {}".format(key, value) for key, value in item["options"].items()))
    content.append({"type": "text", "text": "\n\n".join(text_parts)})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def infer_one(client, item, root, model, parameters, api_key):
    started = time.monotonic()
    raw, error, finish_reason = "", None, None
    try:
        response = client.chat.completions.create(model=model, messages=build_messages(item, root), **parameters)
        if not response.choices or response.choices[0].message is None:
            error = "empty_response"
        else:
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            if finish_reason != "stop":
                error = "incomplete_response"
            else:
                raw = str(choice.message.content or "").strip().replace(api_key, "[REDACTED]")
    except APIStatusError as exc:
        error = "http_{}".format(exc.status_code)
    except Exception as exc:
        error = type(exc).__name__
    parsed = extract_answer(raw, item["options"], item["task_id"])
    if parsed is None and error is None:
        error = "invalid_answer"
    return {
        "question_id": item["question_id"],
        "task_id": item["task_id"],
        "group_id": item.get("group_id"),
        "sub_id": item.get("sub_id"),
        "model": model,
        "ground_truth": item["answer"],
        "model_answer": raw,
        "parsed_answer": parsed,
        "format_ok": parsed is not None,
        "parse_failed": parsed is None,
        "correct": parsed is not None and parsed == normalize_answer(item["answer"], set(item["options"])),
        "finish_reason": finish_reason,
        "error": error,
        "latency_seconds": round(time.monotonic() - started, 3),
    }


def main():
    args = parse_args()
    input_path = args.data_path.expanduser().resolve()
    rows = load_questions(input_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / "results.jsonl"
    config_path = output_dir / "run_config.json"
    if output_path.exists() and not args.resume:
        raise SystemExit("Output already exists. Use --resume or a new --output-dir.")
    parameters = {}
    if args.temperature is not None:
        parameters["temperature"] = args.temperature
    if args.max_tokens is not None:
        parameters["max_tokens"] = args.max_tokens
    run_config = {
        "model": args.model,
        "prompt_version": "single-title-v1",
        "answer_parsing": ANSWER_PARSING_POLICY,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "api_base_sha256": hashlib.sha256(args.api_base.rstrip("/").encode()).hexdigest(),
        "parameters": parameters,
    }
    if args.resume and output_path.exists():
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise SystemExit("Saved dataset, prompt, parser, endpoint, model or generation parameters differ; use a new output directory.")
    by_id = {row["question_id"]: row for row in rows}
    completed = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            qid = result["question_id"]
            if qid not in by_id or qid in completed or result.get("model") != args.model:
                raise SystemExit("Saved results contain unknown, duplicate or incompatible records.")
            for field in ("task_id", "group_id", "sub_id"):
                if result.get(field) != by_id[qid].get(field):
                    raise SystemExit("Saved result identifiers do not match this dataset.")
            if result.get("ground_truth") != by_id[qid]["answer"]:
                raise SystemExit("Saved reference answers do not match this dataset.")
            completed.add(qid)
    selected = [by_id[qid] for qid in SMOKE_IDS] if args.smoke else rows
    pending = [item for item in selected if item["question_id"] not in completed]
    interrupted = False
    if pending:
        api_key = os.environ.get("API_KEY")
        if not api_key or not api_key.strip():
            raise SystemExit("Set API_KEY in your local environment. Never put credentials in source files.")
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        with OpenAI(api_key=api_key, base_url=args.api_base, timeout=300, max_retries=2) as client:
            with output_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures, saved = [], set()

                def save_result(future):
                    handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                    handle.flush()
                    saved.add(future)

                try:
                    for item in pending:
                        futures.append(pool.submit(infer_one, client, item, input_path.parent, args.model, parameters, api_key))
                    for future in tqdm(as_completed(futures), total=len(futures), desc="PURE", unit="question"):
                        save_result(future)
                except KeyboardInterrupt:
                    interrupted = True
                    for future in futures:
                        future.cancel()
                    print("Stopped queued requests; waiting for in-flight responses to save. Use --resume to continue.")
                    for future in as_completed([f for f in futures if not f.cancelled() and f not in saved]):
                        save_result(future)
    metrics, _ = write_metrics(input_path, output_path)
    print_metrics(metrics)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
