"""Continue the SFT LoRA adapter with tool-using GRPO on GSM8K.

Example: uv run --no-project --python .venv/bin/python -- python src/train_rl.py
"""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TRACKIO_DIR", str(ROOT / "results/trackio"))

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from data import MODEL_ID, prep_rl_data
from rewards import format_reward, outcome_reward
from tools import calculator, submit_answer


def train(args):
    set_seed(args.seed)
    train_dataset = prep_rl_data(args.train_data)
    val_dataset = prep_rl_data(args.val_data)
    if args.train_limit is not None:
        train_dataset = train_dataset.select(range(min(args.train_limit, len(train_dataset))))
    if args.val_limit is not None:
        val_dataset = val_dataset.select(range(min(args.val_limit, len(val_dataset))))

    adapter_config = json.loads((args.sft_checkpoint / "adapter_config.json").read_text())
    base_id = adapter_config.get("base_model_name_or_path") or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(str(args.sft_checkpoint))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Materialize the SFT policy once: base weights + trained SFT adapter.
    # This is the frozen starting point for a newly initialized RL LoRA.
    merged_weights = args.merged_sft_dir / "model.safetensors"
    if not merged_weights.exists():
        base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype)
        sft_model = PeftModel.from_pretrained(base_model, str(args.sft_checkpoint))
        merged_model = sft_model.merge_and_unload()
        args.merged_sft_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(args.merged_sft_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(args.merged_sft_dir))
        del sft_model, merged_model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged_model = AutoModelForCausalLM.from_pretrained(str(args.merged_sft_dir), dtype=dtype)
    # Match train_sft.py exactly, but initialize new LoRA weights on the
    # already merged SFT model rather than continuing the SFT adapter.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(merged_model, lora_config)
    lora = model.peft_config[model.active_adapter]
    trainable_parameters, total_parameters = model.get_nb_trainable_parameters()

    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=2,
        num_generations=args.num_generations,
        num_generations_eval=2,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(5, args.max_steps),
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=min(100, args.max_steps),
        eval_strategy="no" if args.skip_validation else "steps",
        eval_steps=None if args.skip_validation else args.max_steps,
        report_to="trackio",
        project=args.trackio_project,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=[outcome_reward, format_reward],
        processing_class=tokenizer,
        tools=[calculator, submit_answer],
    )
    started = time.monotonic()
    outcome = trainer.train()
    validation = next(
        (item for item in reversed(trainer.state.log_history) if "eval_reward" in item),
        None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    result = {
        "sft_checkpoint": str(args.sft_checkpoint),
        "merged_sft_base": str(args.merged_sft_dir),
        "rl_adapter_initialized_fresh": True,
        "rl_checkpoint": str(args.output_dir),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "steps": trainer.state.global_step,
        "training_loss": outcome.training_loss,
        "lora": {
            "r": lora.r,
            "lora_alpha": lora.lora_alpha,
            "lora_dropout": lora.lora_dropout,
            "bias": lora.bias,
            "target_modules": sorted(lora.target_modules),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
        "validation": validation,
        "duration_seconds": round(time.monotonic() - started, 2),
        "config": {
            "seed": args.seed,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "max_tool_calling_iterations": args.max_tool_calling_iterations,
            "learning_rate": args.learning_rate,
        },
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 GSM8K tool use with GRPO")
    parser.add_argument("--sft-checkpoint", type=Path, default=ROOT / "results/checkpoints/sft")
    parser.add_argument("--merged-sft-dir", type=Path, default=ROOT / "results/checkpoints/sft_merged")
    parser.add_argument("--train-data", type=Path, default=ROOT / "data/gsm8k_rl_train_raw.jsonl")
    parser.add_argument("--val-data", type=Path, default=ROOT / "data/gsm8k_val.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/checkpoints/rl")
    parser.add_argument("--metrics-path", type=Path, default=ROOT / "results/rl_training.json")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trackio-project", default="huggingface")
    parser.add_argument("--run-name", default="qwen3-gsm8k-grpo")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.batch_size % args.num_generations:
        parser.error("--batch-size must be divisible by --num-generations")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    result = train(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
