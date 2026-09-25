"""Evaluate an SFT or RL LoRA checkpoint with executable tool-call rollouts."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
)

from data import MODEL_ID, SYSTEM_PROMPT, prep_rl_data
from rewards import score_completion
from tools import calculator, submit_answer

ROOT = Path(__file__).resolve().parent.parent
TOOLS = [calculator, submit_answer]


def load_checkpoint(checkpoint: Path):
    """Load a trained adapter over its SFT base model."""
    if not (checkpoint / "adapter_config.json").exists():
        raise ValueError(f"Expected a LoRA adapter at {checkpoint}")
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    base_id = config.get("base_model_name_or_path") or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    add_response_schema(tokenizer)
    template = None
    if not is_chat_template_prefix_preserving(tokenizer):
        template = get_training_chat_template(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tokenizer, template


@torch.inference_mode()
def rollout(model, tokenizer, template, prompt: list[dict], max_new_tokens: int, max_turns: int, max_completion_tokens: int):
    conversation = list(prompt)
    completion = []
    generated_tokens = 0
    for _ in range(max_turns):
        encoded = tokenizer.apply_chat_template(
            conversation, tools=TOOLS, chat_template=template,
            add_generation_prompt=True, tokenize=True,
            enable_thinking=False, return_dict=True,
        )
        prefix = encoded["input_ids"]
        remaining = max_completion_tokens - generated_tokens
        if remaining <= 0:
            break
        input_ids = torch.tensor([prefix], device=model.device)
        ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=min(max_new_tokens, remaining),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0, len(prefix):].tolist()
        generated_tokens += len(ids)
        if not ids:
            break
        message = parse_response(tokenizer, ids, prefix=prefix)
        if not message:
            break
        conversation.append(message)
        completion.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            function = call.get("function", {}) if call.get("type") == "function" else {}
            name = function.get("name", "unknown")
            arguments = function.get("arguments", {})
            try:
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                if name == "calculator":
                    result = calculator(**arguments)
                elif name == "submit_answer":
                    result = submit_answer(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except (TypeError, ValueError) as exc:
                result = {"error": str(exc)}
            tool_message = {"role": "tool", "name": name, "content": str(result)}
            conversation.append(tool_message)
            completion.append(tool_message)
    return completion, generated_tokens


def log_trackio(summary: dict, project: str, run_name: str) -> None:
    """Log the completed evaluation, including an already saved baseline."""
    os.environ.setdefault("TRACKIO_DIR", str(ROOT / "results/trackio"))
    import trackio

    trackio.init(
        project=project,
        name=run_name,
        config={key: summary[key] for key in ("checkpoint", "data", "examples", "seed", "generation", "system_prompt")},
    )
    try:
        trackio.log({
            f"eval/{key}": value for key, value in summary.items()
            if isinstance(value, (int, float)) and key not in ("seed", "examples")
        }, step=summary["examples"])
    finally:
        trackio.finish()


def evaluate(args):
    set_seed(args.seed)
    dataset = prep_rl_data(args.data)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    model, tokenizer, template = load_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals = {}
    started = time.monotonic()
    with args.predictions.open("w", encoding="utf-8") as file:
        for index, row in enumerate(dataset):
            completion, token_count = rollout(
                model, tokenizer, template, row["prompt"],
                args.max_new_tokens, args.max_turns, args.max_completion_tokens,
            )
            scores = score_completion(completion, row["ground_truth"])
            for name, value in scores.items():
                totals[name] = totals.get(name, 0.0) + value
            record = {
                "index": index,
                "question": row["prompt"][-1]["content"],
                "ground_truth": row["ground_truth"],
                "completion": completion,
                "generated_tokens": token_count,
                "scores": scores,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"Evaluated {index + 1}/{len(dataset)}", flush=True)
    n = len(dataset)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "examples": n,
        "seed": args.seed,
        "system_prompt": SYSTEM_PROMPT,
        "generation": {
            "do_sample": False,
            "max_new_tokens_per_turn": args.max_new_tokens,
            "max_turns": args.max_turns,
            "max_completion_tokens": args.max_completion_tokens,
            "enable_thinking": False,
        },
        "accuracy": totals["accuracy"] / n,
        "mean_reward": (totals["outcome_reward"] + totals["format_reward"]) / n,
        "outcome_reward": totals["outcome_reward"] / n,
        "format_reward": totals["format_reward"] / n,
        "format_compliance": totals["format_compliance"] / n,
        "calculator_use_rate": totals["calculator_used"] / n,
        "one_calculator_per_turn_rate": totals["one_calculator_per_turn"] / n,
        "answer_submission_rate": totals["submitted_answer"] / n,
        "tool_call_success_rate": (
            totals["successful_tool_calls"] / totals["tool_calls"] if totals["tool_calls"] else 0.0
        ),
        "mean_tool_calls": totals["tool_calls"] / n,
        "duration_seconds": round(time.monotonic() - started, 2),
        "predictions": str(args.predictions),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_trackio:
        log_trackio(summary, args.trackio_project, args.trackio_run_name or f"{args.checkpoint.name}-{args.data.stem}-eval")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool-use GSM8K accuracy and rewards")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/checkpoints/sft")
    parser.add_argument("--data", type=Path, default=ROOT / "data/gsm8k_test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/eval_sft_test.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/eval_sft_test.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--trackio-project", default="huggingface")
    parser.add_argument("--trackio-run-name")
    parser.add_argument("--no-trackio", action="store_true")
    parser.add_argument("--log-existing", action="store_true", help="Log an existing evaluation JSON without regenerating")
    args = parser.parse_args()
    if args.log_existing:
        summary = json.loads(args.output.read_text())
        if not args.no_trackio:
            log_trackio(summary, args.trackio_project, args.trackio_run_name or f"{args.checkpoint.name}-{args.data.stem}-eval")
    else:
        summary = evaluate(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
