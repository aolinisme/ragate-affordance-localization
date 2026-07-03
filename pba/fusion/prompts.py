"""Prompt and token helpers for fusion zero-shot evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

__all__ = [
    "build_object_token_candidates",
    "format_object_name",
    "sanitize_token_name",
    "select_token",
]


def format_object_name(obj: str) -> str:
    return obj.replace("_", " ")


def select_token(tokens: List[Dict], candidates: List[str], *, allow_fallback: bool = True) -> Optional[Dict]:
    lowered = [(tok["id"], tok["tok"]) for tok in tokens]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        matches = [entry for entry in lowered if candidate_lower in entry[1].lower()]
        if matches:
            index, token = matches[0]
            return {"index": index, "token": token}
    if not allow_fallback:
        return None
    for index, token in lowered:
        if token not in {"</s>", "▁"}:
            return {"index": index, "token": token}
    return None


def sanitize_token_name(token: str) -> str:
    name = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    if len(name) > 24:
        name = name[:24]
    return name


def build_object_token_candidates(name: str) -> List[str]:
    raw = name.replace("-", " ").strip()
    spaced = raw.replace("_", " ")

    variants: List[str] = []

    def append_unique(values: Iterable[str]) -> None:
        for value in values:
            value = value.strip()
            if value and value not in variants:
                variants.append(value)

    parts = [part.strip() for part in spaced.split() if part.strip()]
    append_unique([spaced, spaced.replace(" ", ""), raw])
    append_unique(parts)

    def singular_forms(text: str) -> Iterable[str]:
        words = [part.strip() for part in text.replace("_", " ").split() if part.strip()]
        forms: List[str] = []

        def add_form(value: str) -> None:
            if value and value not in forms:
                forms.append(value)

        for word in words + [text.strip()]:
            lower = word.lower()
            if len(lower) > 3 and lower.endswith("ies"):
                add_form(word[:-3] + "y")
            if len(lower) > 2 and lower.endswith("es"):
                add_form(word[:-2])
            if len(lower) > 1 and lower.endswith("s"):
                add_form(word[:-1])
        return forms

    for variant in list(variants):
        append_unique(singular_forms(variant))

    candidates: List[str] = []
    seen = set()
    for variant in variants:
        if not variant:
            continue
        forms = [
            variant,
            variant.lower(),
            variant.capitalize(),
            variant.title(),
            variant.replace(" ", ""),
            variant.lower().replace(" ", ""),
        ]
        for form in forms:
            form = form.strip()
            if not form:
                continue
            for candidate in (form, f"▁{form}"):
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
    return candidates
