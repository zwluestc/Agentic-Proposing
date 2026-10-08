from __future__ import annotations

from typing import Any
import os

from agentic_proposing.core.models import ModelTurn, ToolCall
from agentic_proposing.providers.base import TeacherProvider, parse_json_payload
from .data import encode_prompt


def load_model(model_id: str, *, trainable: bool = False):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install training dependencies: python3 -m pip install -e ".[training]"') from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Local 4B training requires a CUDA GPU; API data collection can run without one")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = not trainable
    if trainable:
        model.gradient_checkpointing_enable()
    else:
        model.eval()
        model.requires_grad_(False)
    return model, tokenizer


def action_token_logprobs(model: Any, prompt_ids: list[int], completion_ids: list[int], *,
                          temperature: float = 1.0):
    """Log probabilities of completion tokens; prompt tokens receive no loss."""
    import torch

    if not prompt_ids or not completion_ids:
        raise ValueError("Both prompt and completion tokens are required")
    if temperature <= 0:
        raise ValueError("Log-probability temperature must be positive")
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=model.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    next_logits = logits[:, len(prompt_ids) - 1:-1, :].float() / temperature
    targets = ids[:, len(prompt_ids):]
    return torch.log_softmax(next_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(0).squeeze(-1)


def action_logprob(model: Any, prompt_ids: list[int], completion_ids: list[int]):
    """Joint log probability of one sampled action, before decoding temperature."""
    return action_token_logprobs(model, prompt_ids, completion_ids).sum()


class LocalPolicyProvider(TeacherProvider):
    """HF policy adapter for the existing closed-loop tool harness."""

    def __init__(self, model: Any, tokenizer: Any, *, reference: Any | None = None,
                 max_context_tokens: int = 8192, max_action_tokens: int = 768,
                 record_token_logprobs: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.reference = reference
        self.max_context_tokens = max_context_tokens
        self.max_action_tokens = max_action_tokens
        self.record_token_logprobs = record_token_logprobs

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                 temperature: float = 0.7, max_tokens: int = 768) -> ModelTurn:
        import torch

        action_limit = min(self.max_action_tokens, max_tokens)
        prompt_ids = encode_prompt(self.tokenizer, messages, tools, self.max_context_tokens - action_limit)
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device=self.model.device)
        self.model.eval()
        generation_settings: dict[str, Any] = {
            "input_ids": inputs,
            "attention_mask": torch.ones_like(inputs),
            "max_new_tokens": action_limit,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_settings.update(temperature=temperature, top_p=0.95)
        if self.record_token_logprobs:
            if temperature <= 0:
                raise ValueError("GRPO rollout_temperature must be positive")
            # PPO's behavior probabilities must match the actual sampling policy.
            generation_settings.update(top_p=1.0, top_k=0, min_p=None,
                                       repetition_penalty=1.0)
        self.model.config.use_cache = True
        with torch.no_grad():
            generated = self.model.generate(**generation_settings)
            completion_ids = generated[0, inputs.shape[-1]:].tolist()
            score_temperature = temperature if self.record_token_logprobs else 1.0
            old_tokens = action_token_logprobs(
                self.model, prompt_ids, completion_ids, temperature=score_temperature,
            )
            reference_tokens = action_token_logprobs(
                self.reference, prompt_ids, completion_ids, temperature=score_temperature,
            ) if self.reference is not None else old_tokens
            old_logprob = float(old_tokens.sum().item())
            reference_logprob = float(reference_tokens.sum().item())
            token_metadata = {
                "old_token_logprobs": old_tokens.cpu().tolist(),
                "reference_token_logprobs": reference_tokens.cpu().tolist(),
                "logprob_temperature": score_temperature,
            } if self.record_token_logprobs else {}
        self.model.config.use_cache = False
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        content = decoded
        calls: list[ToolCall] = []
        try:
            payload = parse_json_payload(decoded)
            if isinstance(payload, dict):
                content = str(payload.get("content", ""))
                raw_calls = payload.get("tool_calls", [])
                if isinstance(raw_calls, list):
                    for raw_call in raw_calls:
                        if isinstance(raw_call, dict) and isinstance(raw_call.get("arguments", {}), dict):
                            calls.append(ToolCall(str(raw_call.get("name", "")), dict(raw_call.get("arguments", {}))))
        except ValueError:
            pass
        return ModelTurn(
            content=content,
            tool_calls=calls,
            raw={
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": completion_ids,
                "completion_text": decoded,
                "old_logprob": old_logprob,
                "reference_logprob": reference_logprob,
                **token_metadata,
            },
        )
