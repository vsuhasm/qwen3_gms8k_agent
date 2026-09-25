# Qwen3 GSM8K agent post-training

Train `Qwen/Qwen3-0.6B` to solve GSM8K problems with a calculator and a final-answer tool. The pipeline creates tool-use rollouts for supervised fine-tuning (SFT), then trains a fresh LoRA adapter with GRPO on top of the merged SFT model.

## Installation

From the repository root, with `uv` installed:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install torch 'transformers==5.17.0' 'trl==1.13.0' 'peft==0.21.0' 'datasets==5.0.1' 'accelerate==1.15.0' 'trackio==0.38.1'
```

## Run

The data files in `data/` are ready to use. Run these commands in order:

```bash
uv run src/train_sft.py
uv run src/eval.py --max-completion-tokens 768
uv run src/train_rl.py
uv run src/eval.py \
  --checkpoint results/checkpoints/rl/checkpoint-100 \
  --output results/eval_rl_test.json \
  --predictions results/eval_rl_test.jsonl \
  --trackio-run-name rl-test-eval
```

The first evaluation measures the SFT baseline. The last command evaluates an RL checkpoint; change `checkpoint-100` to the step you want. Each evaluation writes a summary JSON and per-problem completions JSONL. Training and evaluation metrics are logged to Trackio. View them with:

```bash
uv run trackio show
```

## Data

The source is the [GSM8K `main` split](https://huggingface.co/datasets/openai/gsm8k), which has 7,473 training and 1,319 test problems. Sampling uses seed 42; there is no separate validation split in the original dataset.

| Split | Problems | Source |
| --- | ---: | --- |
| Selected training pool | 1,024 (13.7% of original train) | Original train |
| SFT train | 184 raw; 181 processed rollouts | Selected training pool |
| RL train | 737 | Selected training pool |
| Validation | 103 raw; 102 processed SFT rollouts | Selected training pool |
| Test | 256 (19.4% of original test) | Original test |

SFT preprocessing turns GSM8K calculation annotations into conversations for `SFTTrainer`. An abbreviated record from `data/sft_train_proc_rollouts.jsonl` has this shape:

```text
{
  "messages": [
    {"role": "system", "content": "<shared system prompt>"},
    {"role": "user", "content": "<math problem>"},
    {"role": "assistant", "content": "...", "tool_calls": [
      {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "4*20"}}}
    ]},
    {"role": "tool", "name": "calculator", "content": "80"},
    ...,
    {"role": "assistant", "content": "...", "tool_calls": [
      {"type": "function", "function": {"name": "submit_answer", "arguments": {"answer": "108"}}}
    ]},
    {"role": "tool", "name": "submit_answer", "content": "Answer submitted."},
    {"role": "assistant", "content": "Done."}
  ],
  "tools": [<calculator schema>, <submit_answer schema>]
}
```

`prep_rl_data()` converts raw GSM8K rows to `GRPOTrainer` records. The answer is kept out of the prompt and passed to reward functions as `ground_truth`:

```json
{
  "prompt": [
    {"role": "system", "content": "<shared system prompt>"},
    {"role": "user", "content": "What is 8 + 7?"}
  ],
  "ground_truth": "15"
}
```

## Tools and rewards

- `calculator(expression)` evaluates basic arithmetic with numbers, parentheses, `+`, `-`, `*`, and `/`.
- `submit_answer(answer)` accepts a final numeric answer. The model should call it once after its calculations.
- **Outcome reward:** `1.0` for a correct numeric argument in an executed `submit_answer` call; otherwise `0`.
- **Format reward:** `0.25` when a rollout has a valid calculator call, at most one calculator call per assistant turn, exactly one valid answer submission as its last tool call, successful tool responses, and a final assistant message; otherwise `0`.

Evaluation also reports answer accuracy (including plain-text numeric answers), format compliance, calculator use, answer submission, and tool-call success rate. Plain-text accuracy does not earn the outcome reward.

## Training and evaluation settings

| Setting | SFT | RL (GRPO) | Test evaluation |
| --- | --- | --- | --- |
| Model | `Qwen/Qwen3-0.6B` | Merged SFT model + new LoRA | Selected SFT or RL adapter |
| Duration | 1 epoch | 100 optimizer steps by default | 256 test problems |
| Batch / accumulation | 1 / 1 | 4 / 1; 4 generations per prompt | Greedy decoding |
| Learning rate / schedule | `1e-5` / cosine | `1e-5` / cosine | — |
| Warmup | 5 steps | 5 steps | — |
| Length limits | 1,024 tokens per SFT example | 1,024 completion tokens; 8 tool iterations | 192 new tokens per turn; 8 turns; 1,024 completion tokens by default (768 in the saved SFT baseline) |
| LoRA | Rank 16, alpha 16, dropout 0, all linear layers | Same configuration, newly initialized | — |

SFT uses assistant-only loss, evaluates on the processed validation set at the end, and saves its adapter to `results/checkpoints/sft`. RL validates at the final step using two generations per validation prompt, saves checkpoints every 100 steps (or at the final step for shorter runs), and saves its adapter to `results/checkpoints/rl`. Training and evaluation use seed 42 and disable Qwen3 thinking mode.

## SFT results

The saved [SFT training summary](results/sft_training.json) records one epoch over 181 processed training rollouts (181 optimizer steps), with an average training loss of **0.7586** and validation loss of **0.5447** on 102 processed validation rollouts.

The [SFT test evaluation](results/eval_sft_test.json) used greedy decoding, seed 42, and a 768-token completion limit on 256 held-out problems:

| Metric | Result |
| --- | ---: |
| Answer accuracy | **35.5%** (91/256) |
| Mean total reward | 0.2461 |
| Correct `submit_answer` outcome reward | 15.2% (39/256) |
| Format compliance / mean format reward | 37.5% / 0.0938 |
| Calculator use / answer submission | 41.0% / 41.4% |
| Tool-call success rate | 89.1% |

Answer accuracy also counts plain-text numeric answers; the outcome reward requires a correct tool submission. Per-problem completions and scores are in [the SFT predictions file](results/eval_sft_test.jsonl).

| Mean token accuracy | Training loss | Gradient norm |
| :---: | :---: | :---: |
| ![SFT training mean token accuracy](image.png) | ![SFT training loss](image-1.png) | ![SFT training gradient norm](image-2.png) |
