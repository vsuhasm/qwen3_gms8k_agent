"""Tool-aware GRPO rewards and the shared evaluation rubric."""

from collections.abc import Sequence
import re

from tools import InvalidExpression, calculate, parse_number, render_number


def _calls(completion: Sequence[dict]) -> list[tuple[str, dict, str | None]]:
    """Pair assistant tool calls with tool responses in their original order."""
    calls = []
    for index, message in enumerate(completion):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        for offset, call in enumerate(tool_calls):
            function = call.get("function", {}) if call.get("type") == "function" else {}
            name = function.get("name", "")
            arguments = function.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            response = None
            response_index = index + offset + 1
            if response_index < len(completion):
                following = completion[response_index]
                if following.get("role") == "tool" and following.get("name") == name:
                    response = following.get("content")
            calls.append((name, arguments, response))
    return calls


def score_completion(completion: Sequence[dict], ground_truth: str) -> dict[str, float]:
    """Score an executed rollout using only actual tool calls and tool responses."""
    calls = _calls(completion)
    one_calculator_per_turn = all(
        sum(
            call.get("type") == "function" and call.get("function", {}).get("name") == "calculator"
            for call in (message.get("tool_calls") or [])
        ) <= 1
        for message in completion if message.get("role") == "assistant"
    )
    successes = []
    calculator_valid = False
    submitted = []
    for name, arguments, response in calls:
        if name == "calculator" and set(arguments) == {"expression"}:
            expression = arguments["expression"]
            try:
                result = render_number(calculate(expression))
                valid = response == result
            except (InvalidExpression, TypeError):
                valid = False
            calculator_valid |= valid
        elif name == "submit_answer" and set(arguments) == {"answer"}:
            try:
                answer = parse_number(arguments["answer"])
                valid = response == "Answer submitted."
                if valid:
                    submitted.append(answer)
            except (InvalidExpression, TypeError, AttributeError):
                valid = False
        else:
            valid = False
        successes.append(valid)
    try:
        target = parse_number(ground_truth)
    except (InvalidExpression, TypeError, AttributeError):
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    correct_submission = any(answer == target for answer in submitted)
    # Answer accuracy also counts a plain-text final number for diagnostic comparison.
    # The training outcome reward below still requires an executed submission.
    text_answer = None
    if not submitted:
        assistant_text = " ".join(
            str(message.get("content") or "")
            for message in completion if message.get("role") == "assistant"
        )
        numbers = re.findall(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])", assistant_text)
        if numbers:
            try:
                text_answer = parse_number(numbers[-1])
            except InvalidExpression:
                pass
    correct = correct_submission or text_answer == target
    final_submit = bool(calls) and calls[-1][0] == "submit_answer"
    format_ok = (
        calculator_valid and one_calculator_per_turn and len(submitted) == 1 and final_submit
        and bool(successes) and all(successes)
        and len(completion) > 0 and completion[-1].get("role") == "assistant"
        and not completion[-1].get("tool_calls")
    )
    return {
        "accuracy": float(correct),
        "outcome_reward": float(correct_submission),
        "format_reward": 0.25 if format_ok else 0.0,
        "format_compliance": float(format_ok),
        "calculator_used": float(calculator_valid),
        "one_calculator_per_turn": float(one_calculator_per_turn),
        "submitted_answer": float(bool(submitted)),
        "tool_calls": float(len(calls)),
        "successful_tool_calls": float(sum(successes)),
        "tool_call_success_rate": sum(successes) / len(calls) if calls else 0.0,
    }


def outcome_reward(completions, ground_truth, **kwargs) -> list[float]:
    """Reward a correct numeric argument to an executed submit_answer call."""
    return [score_completion(c, gt)["outcome_reward"] for c, gt in zip(completions, ground_truth, strict=True)]


def format_reward(completions, ground_truth, **kwargs) -> list[float]:
    """Reward a valid calculator call followed by a single final submission."""
    return [score_completion(c, gt)["format_reward"] for c, gt in zip(completions, ground_truth, strict=True)]
