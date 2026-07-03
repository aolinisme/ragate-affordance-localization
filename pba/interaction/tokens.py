"""Token matching helpers for interaction probing."""

from __future__ import annotations

from typing import Dict

__all__ = ["collect_token_index"]


def collect_token_index(tokenizer, prompt: str, affordance: str) -> Dict[str, int]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
    wanted = affordance.lower()
    for index, token in enumerate(tokens):
        key = token.lower()
        if key == wanted or wanted in key:
            return {affordance: index}
    return {}
