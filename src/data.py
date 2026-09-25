import json
import random
import re
from statistics import median
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import TOOL_SCHEMAS, InvalidExpression as InvalidExample, calculate, parse_number, render_number



MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "openai/gsm8k"
RANDOM_SEED = 42
MAX_ROLLOUT_LENGTH = 1024
DATA_DIR = "/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/data"
SYSTEM_PROMPT = """As a faithful assistant follow this workflow:
- Break down the problem into simple intermediate steps
- At each step, invoke calculator tool for any arithmetic computation. Use the tool output provided to you in your next steps.
- Once you arrive at your final answer in the end, invoke submit_answer tool with the final numerical answer and terminate.
"""
ANNOTATION = re.compile(r"<<([^<>]+?)=([^<>]+?)>>")


def prep_gsm8k_subset_data():
    # dataset split params
    train_dataset_size = 1024
    train_split_prop = 0.9
    train_split_size = int(train_split_prop * train_dataset_size)
    val_split_size = train_dataset_size - train_split_size
    test_dataset_size = 256
    sft_train_split_prop = 0.2
    sft_train_split_size = int(sft_train_split_prop * train_split_size)
    rl_train_split_size = train_split_size - sft_train_split_size

    print(f"Subset train size: {train_dataset_size}")
    print(f"Subset val size: {val_split_size}")
    print(f"Subset test size: {test_dataset_size}")
    print(f"SFT train size: {sft_train_split_size}")
    print(f"RL train size: {rl_train_split_size}")

    # load gsm8k dataset from hf hub
    train_dataset = load_dataset(DATASET_ID, "main", split="train")
    test_dataset = load_dataset(DATASET_ID, "main", split="test")

    print(f"Full dataset train size: {len(train_dataset)}")
    print(f"Full dataset test size: {len(test_dataset)}")

    # get sample ids, shuffle 
    rng = random.Random(RANDOM_SEED)
    train_ids = list(range(len(train_dataset)))
    test_ids = list(range(len(test_dataset)))
    rng.shuffle(train_ids)
    rng.shuffle(test_ids)

    splits = {
        "sft_train_raw": train_dataset.select(train_ids[:sft_train_split_size]),
        "rl_train_raw": train_dataset.select(train_ids[sft_train_split_size:train_split_size]),
        "val": train_dataset.select(train_ids[train_split_size:train_dataset_size]),
        "test": test_dataset.select(test_ids[:test_dataset_size]),
    }
    for name, dataset in splits.items():
        dataset.to_json(f"{DATA_DIR}/gsm8k_{name}.jsonl")

def load_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

def write_jsonl(file_path: str, rows: list[dict]) -> None:
    with open(file_path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

def tool_call(name: str, arguments: dict[str, str], content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }

def format_sample(question: str, answer: str) -> tuple[dict, dict]:
    """Preserve original reasoning segments and interleave verified tool results."""
    solution, delimiter, final_text = answer.rpartition("####")
    final_answer = render_number(parse_number(final_text))
    matches = list(ANNOTATION.finditer(solution))
    if len(matches) < 1:
        print(solution)
        raise ValueError("no calculator calls observed")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps = []
    cursor = 0
    for match in matches:
        expression = match.group(1).strip()
        actual = calculate(expression)
        reasoning = solution[cursor:match.start()]
        messages.append(tool_call("calculator", {"expression": expression}, content=reasoning))
        result = render_number(actual)
        messages.append({"role": "tool", "name": "calculator", "content": result})
        steps.append({"expression": expression, "result": result})
        cursor = match.end()

    tail = "Final answer: " + solution[cursor:]
    messages.append(tool_call("submit_answer", {"answer": final_answer}, content=tail))
    messages.append({"role": "tool", "name": "submit_answer", "content": "Answer submitted."})
    messages.append({"role": "assistant", "content": "Done."})
    record = {"messages": messages, "tools": TOOL_SCHEMAS}
    audit = {"question": question, "solution": solution, "final_answer": final_answer, "steps": steps}
    return record, audit

def process_sft_jsonl(tokenizer, sft_train_data, output_name):
    train_num_tokens = []
    num_skipped = 0
    train_samples = []
    for sample in sft_train_data:
        try:
            record, audit = format_sample(sample["question"], sample["answer"])
            tokenized = tokenizer.apply_chat_template(
                record["messages"],
                tools=TOOL_SCHEMAS,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            token_count = len(tokenized["input_ids"])
            train_num_tokens.append(token_count)
            train_samples.append((record, audit, token_count))
        except:
            num_skipped += 1
            print("skipped sample")

    seq_lens = sorted(train_num_tokens)
    percentile = lambda p: seq_lens[round((len(seq_lens) - 1) * p)]
    token_lengths = {"min": seq_lens[0], "median": median(seq_lens), "p95": percentile(.95),
                                "p99": percentile(.99), "max": seq_lens[-1]}
    print("token lengths", token_lengths)
    print("tokens total", sum(seq_lens))
    print("num samples skipped", num_skipped)
    write_jsonl(f"{DATA_DIR}/{output_name}.jsonl", [item[0] for item in train_samples])

def prep_sft_data():
    sft_train_path = "/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/data/gsm8k_sft_train_raw.jsonl"
    sft_val_path = "/home/suhas/Documents/code/qwen3_gsm8k_agent_posttraining/data/gsm8k_val.jsonl"

    sft_train_data = load_jsonl(sft_train_path)
    sft_val_data = load_jsonl(sft_val_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    process_sft_jsonl(tokenizer, sft_train_data, "sft_train_proc_rollouts")
    process_sft_jsonl(tokenizer, sft_val_data, "sft_val_proc_rollouts")
    

def prep_rl_data(file_path: str) -> Dataset:
    """Build conversational GRPO prompts with reference answers kept out of the prompt.

    TRL forwards the ``ground_truth`` column to custom reward functions.
    Evaluation uses this same prompt construction.
    """
    rows = []
    for index, sample in enumerate(load_jsonl(str(file_path)), start=1):
        try:
            answer = sample["answer"].rpartition("####")[2]
            if not answer:
                raise ValueError("missing #### final answer")
            ground_truth = render_number(parse_number(answer))
            question = sample["question"].strip()
            if not question:
                raise ValueError("empty question")
        except (KeyError, InvalidExample, ValueError) as exc:
            raise ValueError(f"Invalid RL example at {file_path}:{index}: {exc}") from exc
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ground_truth": ground_truth,
        })
    if not rows:
        raise ValueError(f"No examples in {file_path}")
    return Dataset.from_list(rows)

if __name__ == "__main__":
    prep_sft_data()
