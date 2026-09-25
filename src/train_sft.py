import argparse
import json
import platform
import time
from pathlib import Path

import datasets
import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from data import load_jsonl, MODEL_ID


def load_records(file_path: str) -> Dataset:
    rows = load_jsonl(file_path)
    if not rows:
        raise ValueError(f"No examples in {file_path}")
    # TRL forwards this field to Qwen's chat template for every conversation.
    for row in rows:
        row["chat_template_kwargs"] = {"enable_thinking": False}
    return Dataset.from_list(rows, on_mixed_types="use_json")


def enable_assistant_masks(tokenizer) -> None:
    # Qwen's default template has no generation blocks, which TRL needs for assistant_only_loss.
    template = tokenizer.chat_template
    assistant_start = '{%- elif message.role == "assistant" %}'
    assistant_close = r"{{- '<|im_end|>\n' }}"
    tool_branch = '{%- elif message.role == "tool" %}'
    assistant_end = assistant_close + "\n    " + tool_branch
    if template.count(assistant_start) != 1 or template.count(assistant_end) != 1:
        raise ValueError("Unexpected Qwen chat template; cannot mark assistant tokens")
    tokenizer.chat_template = (
        template.replace(assistant_start, assistant_start + "{% generation %}", 1)
        .replace(assistant_end, assistant_close + "{% endgeneration %}\n    " + tool_branch, 1)
    )


def train(args: argparse.Namespace) -> dict:
    start = time.monotonic()
    set_seed(args.seed)
    training = load_records(args.data_dir / "sft_train_proc_rollouts.jsonl")
    validation = load_records(args.data_dir / "sft_val_proc_rollouts.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        tf32=True,
        max_length=1024,
        assistant_only_loss=True,
        eos_token="<|im_end|>",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="trackio",
        seed=args.seed,
        data_seed=args.seed,
    )
    lora = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=training,
        eval_dataset=validation,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    outcome = trainer.train()
    eval_metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    result = {
        "model": MODEL_ID,
        "train_examples": len(training),
        "validation_examples": len(validation),
        "seed": args.seed,
        "optimizer_steps": trainer.state.global_step,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_loss": outcome.training_loss,
        "validation_loss": eval_metrics["eval_loss"],
        "duration_seconds": round(time.monotonic() - start, 2),
        "log_history": trainer.state.log_history,
        "adapter_dir": str(args.output_dir),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT trainer")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/results/checkpoints/sft"))
    parser.add_argument("--metrics-path", type=Path, default=Path("/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/results/sft_training.json"))
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(args)
    print(json.dumps({key: result[key] for key in (
        "optimizer_steps", "training_loss", "validation_loss", "duration_seconds", "adapter_dir"
    )}, indent=2))


if __name__ == "__main__":
    main()
