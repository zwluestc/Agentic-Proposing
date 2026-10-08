from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any
import re


def extract_boxed(text: str) -> str | None:
    """Return the last balanced LaTeX boxed expression, including nested braces."""
    result = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    result = text[start:index]
                    break
    return result


def _answer_text(value: Any) -> str:
    text = str(value).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed
    text = re.sub(r"^(?:final answer|answer)\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().strip("$ .。")
    return "".join(text.casefold().split())


def answers_match(predicted: Any, expected: Any) -> bool:
    left, right = _answer_text(predicted), _answer_text(expected)
    if left == right and bool(left):
        return True
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        pass
    try:
        return Fraction(left) == Fraction(right)
    except (ValueError, ZeroDivisionError):
        pass
    fraction = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")
    left_match, right_match = fraction.fullmatch(left), fraction.fullmatch(right)
    try:
        left_value = Fraction(int(left_match[1]), int(left_match[2])) if left_match else Fraction(left)
        right_value = Fraction(int(right_match[1]), int(right_match[2])) if right_match else Fraction(right)
        return left_value == right_value
    except (ValueError, ZeroDivisionError):
        return False


class FrozenProber:
    """Untrained solver used for the paper's Pass@16 difficulty estimate."""

    def __init__(self, model: Any, tokenizer: Any, *, attempts: int = 16,
                 temperature: float = 1.4, max_new_tokens: int = 256):
        if attempts < 1:
            raise ValueError("Prober attempts must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.attempts = attempts
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def pass_rate(self, task: dict[str, Any]) -> float:
        import torch

        if isinstance(task.get("answer"), (dict, list)):
            raise ValueError("The built-in rule grader requires a scalar task answer")
        if str(task.get("task_type", "")) in {"coding", "code", "programming"}:
            raise ValueError("Coding tasks require a configured execution-based grader")
        question = str(task["question"])
        if task.get("task_type") == "single_choice" and task.get("metadata", {}).get("options"):
            options = task["metadata"]["options"]
            question += "\nOptions: " + "; ".join(
                f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
            )
        messages = [
            {"role": "system", "content": "Solve the problem independently. Put exactly one final answer in \\boxed{...}."},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
        ).to(self.model.device)
        correct = 0
        gold = task["answer"]
        options = task.get("metadata", {}).get("options", [])
        accepted = [gold]
        if task.get("task_type") == "single_choice" and isinstance(options, list):
            if isinstance(gold, str) and len(gold.strip()) == 1 and gold.strip().upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]:
                accepted.append(options[ord(gold.strip().upper()) - 65])
            else:
                accepted.extend(
                    chr(65 + index) for index, option in enumerate(options)
                    if answers_match(option, gold)
                )
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.attempts):
                output = self.model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(output[0, prompt.shape[-1]:], skip_special_tokens=True)
                correct += any(answers_match(response, value) for value in accepted)
        return correct / self.attempts
