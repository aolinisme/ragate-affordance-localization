from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "geometry_probing" / "umd_linear_probing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pba.geometry.config import load_config  # noqa: E402
from src.engine.trainer import LinearProbeExperiment  # noqa: E402


def _json_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def export_teacher_logits(
    *,
    checkpoint: Path,
    config_path: Path,
    split: str,
    output_dir: Path,
    local_override: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, local_override)
    experiment = LinearProbeExperiment(config)

    head = experiment._build_head().to(experiment.device)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    head.load_state_dict(checkpoint_payload.get("state_dict", checkpoint_payload))
    head.eval()
    experiment.backbone.eval()

    loader = {
        "train": experiment.train_loader,
        "val": experiment.val_loader,
        "test": experiment.test_loader,
    }[split]
    precision = experiment.training_cfg.get("precision", "bf16")
    logits_dir = output_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(experiment.device, non_blocking=True)
            features = experiment.backbone(images, autocast_precision=precision)
            features = experiment._merge_geometry_features(features, batch)
            logits = experiment._apply_head(head, features).detach().cpu()
            for idx, meta in enumerate(batch["meta"]):
                frame_id = str(meta["frame_id"])
                out_path = logits_dir / f"{frame_id}.pt"
                torch.save(logits[idx].to(torch.float16), out_path)
                entries.append(
                    {
                        "frame_id": frame_id,
                        "tool": str(meta.get("tool", "")),
                        "path": str(out_path.relative_to(output_dir)),
                        "shape": list(logits[idx].shape),
                        "meta": _json_safe_meta(meta),
                    }
                )

    manifest = {
        "evidence_tier": "teacher_logits",
        "teacher_checkpoint": str(checkpoint.resolve()),
        "teacher_config": str(config_path.resolve()),
        "split": split,
        "num_entries": len(entries),
        "entries": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{split}_teacher_logits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "num_entries": len(entries)}, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export patch-level teacher logits for lightweight student distillation.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    export_teacher_logits(
        checkpoint=args.checkpoint,
        config_path=args.config,
        split=args.split,
        output_dir=args.output_dir,
        local_override=args.local,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
