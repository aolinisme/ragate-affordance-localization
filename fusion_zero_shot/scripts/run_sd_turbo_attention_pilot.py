#!/usr/bin/env python3
"""Run a lightweight SD-Turbo attention pilot on AGD20K.

The goal is to test whether a much cheaper diffusion model can provide a
verb-conditioned spatial prior that is useful enough to become a student or
fallback interaction branch for the heavier FLUX pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pba.data.affordance import iter_agd20k_samples  # noqa: E402
from pba.metrics.affordance import cal_kl, cal_nss, cal_sim  # noqa: E402


@dataclass
class AttentionStore:
    maps: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.maps.clear()


class RecordingAttnProcessor:
    """Diffusers attention processor that stores cross-attention probabilities."""

    def __init__(self, store: AttentionStore, max_tokens: int = 77) -> None:
        self.store = store
        self.max_tokens = max_tokens

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        spatial_hw: tuple[int, int] | None = None
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            spatial_hw = (height, width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross and attention_probs.shape[-1] <= self.max_tokens:
            heads = max(1, attention_probs.shape[0] // batch_size)
            probs = attention_probs.detach().float().view(batch_size, heads, -1, attention_probs.shape[-1])
            self.store.maps.append(probs.mean(dim=1).cpu())

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4 and spatial_hw is not None:
            height, width = spatial_hw
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def _prompt_for(affordance: str, object_name: str) -> str:
    verb = affordance.replace("_", " ")
    obj = object_name.replace("_", " ")
    return f"a hand {verb} {obj}"


def _token_indices(pipe: StableDiffusionPipeline, prompt: str, needle: str) -> tuple[list[int], list[str]]:
    tokens = pipe.tokenizer.tokenize(prompt)
    ids: list[int] = []
    target_parts = needle.replace("_", " ").split()
    for target in target_parts:
        target_clean = target.lower()
        for i, token in enumerate(tokens):
            clean = token.replace("</w>", "").lower()
            if clean == target_clean or target_clean in clean:
                ids.append(i + 1)
    if not ids:
        ids = [1]
    return sorted(set(ids)), tokens


def _attention_heatmap(store: AttentionStore, token_indices: list[int], target_hw: tuple[int, int]) -> np.ndarray:
    candidates: list[torch.Tensor] = []
    for attn in store.maps:
        if attn.ndim != 3 or attn.shape[0] < 1:
            continue
        spatial = attn.shape[1]
        side = int(round(math.sqrt(spatial)))
        if side * side != spatial:
            continue
        valid_tokens = [idx for idx in token_indices if idx < attn.shape[-1]]
        if not valid_tokens:
            continue
        heat = attn[0, :, valid_tokens].mean(dim=-1).view(1, 1, side, side)
        candidates.append(heat)
    if not candidates:
        return np.zeros(target_hw, dtype=np.float32)

    resized = [F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False) for h in candidates]
    heat = torch.stack(resized, dim=0).mean(dim=0)[0, 0]
    arr = heat.numpy().astype(np.float32)
    arr = arr - float(arr.min())
    denom = float(arr.max())
    if denom > 1e-8:
        arr = arr / denom
    return arr


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def _save_heat(path: Path, heat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _heatmap_stats(arr: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    flat = values.reshape(-1)
    if flat.size == 0:
        return {"entropy": 0.0, "top5_mass": 0.0, "peak": 0.0, "mean": 0.0}
    total = float(flat.sum())
    if total <= eps:
        return {"entropy": 0.0, "top5_mass": 0.0, "peak": float(flat.max()), "mean": float(flat.mean())}
    prob = flat / total
    entropy = -float(np.sum(prob * np.log(prob + eps))) / math.log(float(flat.size))
    k = max(1, int(math.ceil(flat.size * 0.05)))
    top = np.partition(prob, -k)[-k:]
    return {
        "entropy": entropy,
        "top5_mass": float(top.sum()),
        "peak": float(flat.max()),
        "mean": float(flat.mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_keys_from_summary(path: Path) -> set[tuple[str, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        return {
            (row["affordance"], row["object"], row["image"])
            for row in rows
            if row.get("affordance") and row.get("object") and row.get("image")
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/sd-turbo"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sd_turbo_attention_pilot"))
    parser.add_argument(
        "--sample-summary",
        type=Path,
        default=None,
        help="Optional existing AGD20K summary.csv; run only matching affordance/object/image rows.",
    )
    parser.add_argument("--affordances", nargs="+", default=["hold", "cut", "drink_with"])
    parser.add_argument("--max-images-per-object", type=int, default=1)
    parser.add_argument("--max-samples-total", type=int, default=5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.output_root / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_dir.resolve()),
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    load_seconds = time.time() - t_load

    store = AttentionStore()
    processors = {name: RecordingAttnProcessor(store) for name in pipe.unet.attn_processors}
    pipe.unet.set_attn_processor(processors)

    rows: list[dict[str, Any]] = []
    samples = list(iter_agd20k_samples(args.dataset_root, args.affordances, args.max_images_per_object))
    if args.sample_summary is not None:
        keys = _sample_keys_from_summary(args.sample_summary)
        samples = [
            sample
            for sample in samples
            if (sample.affordance, sample.object_name, sample.image_path.name) in keys
        ]
    samples = samples[: args.max_samples_total]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for sample in samples:
        prompt = _prompt_for(sample.affordance, sample.object_name)
        token_ids, tokens = _token_indices(pipe, prompt, sample.affordance)
        object_token_ids, _ = _token_indices(pipe, prompt, sample.object_name)
        store.clear()
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]
        infer_seconds = time.time() - t0

        stem = f"{sample.affordance}_{sample.object_name}_{sample.image_path.stem}"
        sample_dir = out_dir / sample.affordance / sample.object_name / sample.image_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        image.save(sample_dir / "generated.png")
        heat = _attention_heatmap(store, token_ids, (args.height, args.width))
        object_heat = _attention_heatmap(store, object_token_ids, (args.height, args.width))
        _save_heat(sample_dir / "verb_attention.png", heat)
        _save_heat(sample_dir / "object_attention.png", object_heat)
        np.save(sample_dir / "verb_attention.npy", heat)
        np.save(sample_dir / "object_attention.npy", object_heat)
        gt = _load_gray(sample.gt_path, (args.height, args.width))
        verb_stats = _heatmap_stats(heat)
        object_stats = _heatmap_stats(object_heat)
        row = {
            "affordance": sample.affordance,
            "object": sample.object_name,
            "image": sample.image_path.name,
            "prompt": prompt,
            "token_indices": json.dumps(token_ids),
            "object_token_indices": json.dumps(object_token_ids),
            "tokens": json.dumps(tokens, ensure_ascii=False),
            "heatmap": str(sample_dir / "verb_attention.png"),
            "object_heatmap": str(sample_dir / "object_attention.png"),
            "generated": str(sample_dir / "generated.png"),
            "mKLD": float(cal_kl(heat, gt)),
            "mSIM": float(cal_sim(heat, gt)),
            "mNSS": float(cal_nss(heat, gt)),
            "infer_seconds": infer_seconds,
            "num_attention_maps": len(store.maps),
            "verb_entropy": verb_stats["entropy"],
            "verb_top5_mass": verb_stats["top5_mass"],
            "object_entropy": object_stats["entropy"],
            "object_top5_mass": object_stats["top5_mass"],
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("affordance", "object", "image", "mKLD", "mSIM", "mNSS", "infer_seconds", "num_attention_maps")}))

    _write_csv(out_dir / "summary.csv", rows)
    metrics = {
        "samples": len(rows),
        "load_seconds": load_seconds,
        "mean_infer_seconds": float(np.mean([r["infer_seconds"] for r in rows])) if rows else 0.0,
        "mean_mKLD": float(np.mean([r["mKLD"] for r in rows])) if rows else 0.0,
        "mean_mSIM": float(np.mean([r["mSIM"] for r in rows])) if rows else 0.0,
        "mean_mNSS": float(np.mean([r["mNSS"] for r in rows])) if rows else 0.0,
        "summary_csv": str(out_dir / "summary.csv"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
