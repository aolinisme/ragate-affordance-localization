from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, base: Path) -> Path:
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    return (base / out).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def _robust_depth_normalize(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros_like(depth, dtype=np.float32)
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    if hi <= lo + 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 2.0 - 1.0).astype(np.float32)


def _pad_to_multiple(image: np.ndarray, mask: np.ndarray, multiple: int, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    if new_h == h and new_w == w:
        return image, mask
    pad_h = new_h - h
    pad_w = new_w - w
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=ignore_index)
    return image, mask


class StudentUMDDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_root: Path,
        records: List[Dict[str, str]],
        geom_manifest: Path | None,
        teacher_manifest: Path | None,
        image_size: tuple[int, int],
        ignore_index: int,
    ) -> None:
        self.dataset_root = dataset_root
        self.records = records
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.geom_index: Dict[str, str] = {}
        if geom_manifest is not None and geom_manifest.exists():
            data = json.loads(geom_manifest.read_text(encoding="utf-8"))
            entries: list[dict[str, Any]] = []
            if isinstance(data, dict):
                for key in ("train", "val", "test", "data"):
                    value = data.get(key)
                    if isinstance(value, list):
                        entries.extend(value)
            elif isinstance(data, list):
                entries = data
            for item in entries:
                frame_id = item.get("frame_id")
                depth = item.get("pred_depth_npy")
                if isinstance(frame_id, str) and isinstance(depth, str):
                    self.geom_index[frame_id] = depth
        self.depth_available_count = 0
        self.depth_missing_count = 0
        for record in self.records:
            if self._resolve_depth_path(self.geom_index.get(record["frame_id"])).exists():
                self.depth_available_count += 1
            else:
                self.depth_missing_count += 1
        self.teacher_index: Dict[str, Path] = {}
        self.teacher_shape: tuple[int, int, int] | None = None
        if teacher_manifest is not None and teacher_manifest.exists():
            data = json.loads(teacher_manifest.read_text(encoding="utf-8"))
            manifest_dir = teacher_manifest.parent
            entries = data.get("entries", []) if isinstance(data, dict) else []
            for item in entries:
                frame_id = item.get("frame_id")
                path_value = item.get("path")
                shape_value = item.get("shape")
                if isinstance(frame_id, str) and isinstance(path_value, str):
                    path = Path(path_value)
                    self.teacher_index[frame_id] = path if path.is_absolute() else manifest_dir / path
                if self.teacher_shape is None and isinstance(shape_value, list) and len(shape_value) == 3:
                    self.teacher_shape = tuple(int(x) for x in shape_value)
        self.teacher_available_count = 0
        self.teacher_missing_count = 0
        for record in self.records:
            teacher_path = self.teacher_index.get(record["frame_id"])
            if teacher_path is not None and teacher_path.exists():
                self.teacher_available_count += 1
            else:
                self.teacher_missing_count += 1

    def _resolve_depth_path(self, depth_rel: str | None) -> Path:
        if not depth_rel:
            return Path("__missing_depth_asset__")
        depth_path = Path(depth_rel)
        if depth_path.is_absolute():
            return depth_path
        return self.dataset_root / depth_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        rgb_path = self.dataset_root / record["rgb"]
        label_path = self.dataset_root / record["label_mat"]
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(sio.loadmat(label_path)["gt_label"], dtype=np.int64)
        rgb, mask = _pad_to_multiple(rgb, mask, multiple=32, ignore_index=self.ignore_index)

        h_out, w_out = self.image_size
        rgb = cv2.resize(rgb, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

        depth = np.zeros((h_out, w_out), dtype=np.float32)
        depth_rel = self.geom_index.get(record["frame_id"])
        if depth_rel:
            depth_path = self._resolve_depth_path(depth_rel)
            if depth_path.exists():
                raw_depth = np.load(depth_path).astype(np.float32)
                raw_depth = cv2.resize(raw_depth, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
                depth = _robust_depth_normalize(raw_depth)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32)
        rgb_t = (rgb_t - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        depth_t = torch.from_numpy(depth[None, ...]).to(torch.float32)
        mask_t = torch.from_numpy(mask.astype(np.int64))
        sample = {
            "rgb": rgb_t,
            "depth": depth_t,
            "mask": mask_t,
            "meta": {"frame_id": record["frame_id"], "tool": record["tool"]},
        }
        if self.teacher_shape is not None:
            teacher_logits = torch.zeros(self.teacher_shape, dtype=torch.float32)
            teacher_available = torch.tensor(False)
            teacher_path = self.teacher_index.get(record["frame_id"])
            if teacher_path is not None and teacher_path.exists():
                loaded = torch.load(teacher_path, map_location="cpu")
                teacher_logits = loaded.to(torch.float32)
                teacher_available = torch.tensor(True)
            sample["teacher_logits"] = teacher_logits
            sample["teacher_available"] = teacher_available
        return sample


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReliabilityGatedStudent(nn.Module):
    def __init__(self, num_classes: int, width: int = 24, fusion_mode: str = "gate") -> None:
        super().__init__()
        if fusion_mode not in {"rgb", "depth", "concat", "gate", "depth_modulated_rgb"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.rgb = (
            nn.Sequential(
                ConvBlock(3, width, stride=2),
                ConvBlock(width, width * 2, stride=2),
                ConvBlock(width * 2, width * 3, stride=2),
            )
            if fusion_mode in {"rgb", "concat", "gate", "depth_modulated_rgb"}
            else None
        )
        self.depth = (
            nn.Sequential(
                ConvBlock(1, width // 2, stride=2),
                ConvBlock(width // 2, width, stride=2),
                ConvBlock(width, width * 3, stride=2),
            )
            if fusion_mode in {"depth", "concat", "gate", "depth_modulated_rgb"}
            else None
        )
        fused_ch = width * 3
        self.concat_project = (
            nn.Sequential(
                nn.Conv2d(fused_ch * 2, fused_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(fused_ch),
                nn.ReLU(inplace=True),
            )
            if fusion_mode == "concat"
            else None
        )
        self.gate = (
            nn.Sequential(
                nn.Conv2d(fused_ch * 2, fused_ch, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(fused_ch, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            if fusion_mode == "gate"
            else None
        )
        self.depth_modulator = nn.Conv2d(fused_ch, fused_ch, kernel_size=1) if fusion_mode == "depth_modulated_rgb" else None
        if self.depth_modulator is not None:
            nn.init.zeros_(self.depth_modulator.weight)
            nn.init.zeros_(self.depth_modulator.bias)
        self.decoder = nn.Sequential(
            ConvBlock(fused_ch, width * 2),
            nn.Conv2d(width * 2, num_classes, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "rgb":
            if self.rgb is None:
                raise RuntimeError("RGB encoder is not initialized.")
            fused = self.rgb(rgb)
        elif self.fusion_mode == "depth":
            if self.depth is None:
                raise RuntimeError("Depth encoder is not initialized.")
            fused = self.depth(depth)
        else:
            if self.rgb is None or self.depth is None:
                raise RuntimeError("RGB-D fusion requires both encoders.")
            rgb_feat = self.rgb(rgb)
            depth_feat = self.depth(depth)
            if self.fusion_mode == "concat":
                if self.concat_project is None:
                    raise RuntimeError("Concat projection is not initialized.")
                fused = self.concat_project(torch.cat([rgb_feat, depth_feat], dim=1))
            elif self.fusion_mode == "depth_modulated_rgb":
                if self.depth_modulator is None:
                    raise RuntimeError("Depth modulator is not initialized.")
                modulation = 0.5 * torch.tanh(self.depth_modulator(depth_feat))
                fused = rgb_feat * (1.0 + modulation)
            else:
                if self.gate is None:
                    raise RuntimeError("Reliability gate is not initialized.")
                gate = self.gate(torch.cat([rgb_feat, depth_feat], dim=1))
                fused = rgb_feat * gate + depth_feat * (1.0 - gate)
        logits = self.decoder(fused)
        return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    keep = (target >= 0) & (target < num_classes)
    pred = pred[keep]
    target = target[keep]
    idx = target * num_classes + pred
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).double()


def _miou(confmat: torch.Tensor, ignore_indices: Iterable[int]) -> tuple[float, list[float]]:
    tp = torch.diag(confmat)
    fp = confmat.sum(0) - tp
    fn = confmat.sum(1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.full_like(denom, float("nan")))
    for idx in ignore_indices:
        if 0 <= idx < iou.numel():
            iou[idx] = torch.nan
    return float(torch.nanmean(iou).item()), [float(x) if not torch.isnan(x) else float("nan") for x in iou]


def _estimate_class_weights(
    *,
    dataset_root: Path,
    records: List[Dict[str, str]],
    num_classes: int,
    ignore_index: int,
    ignore_indices: Iterable[int],
    power: float,
    max_weight: float,
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for record in records:
        mask = np.asarray(sio.loadmat(dataset_root / record["label_mat"])["gt_label"], dtype=np.int64)
        valid = mask != ignore_index
        mask = mask[valid]
        keep = (mask >= 0) & (mask < num_classes)
        labels, label_counts = np.unique(mask[keep], return_counts=True)
        counts[labels] += label_counts

    weights = np.ones(num_classes, dtype=np.float32)
    trainable = counts > 0
    for idx in ignore_indices:
        if 0 <= int(idx) < num_classes:
            trainable[int(idx)] = False
    if np.any(trainable):
        ref = np.median(counts[trainable])
        raw = np.power(ref / np.maximum(counts, 1.0), power)
        raw = np.clip(raw, 0.0, max_weight).astype(np.float32)
        weights[trainable] = raw[trainable]
    return torch.from_numpy(weights)


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_available: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
    foreground_only: bool,
    ignore_index: int,
) -> torch.Tensor:
    if teacher_logits.numel() == 0:
        return student_logits.new_tensor(0.0)
    student_patch = F.interpolate(
        student_logits,
        size=teacher_logits.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    temp = max(float(temperature), 1e-6)
    kd = F.kl_div(
        F.log_softmax(student_patch / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1),
        reduction="none",
    ).sum(dim=1)
    valid = teacher_available.to(device=student_logits.device, dtype=torch.bool)[:, None, None]
    if foreground_only:
        target_patch = F.interpolate(
            target[:, None].to(torch.float32),
            size=teacher_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1).to(torch.long)
        valid = valid & (target_patch != 0) & (target_patch != ignore_index)
    valid_f = valid.to(kd.dtype)
    denom = valid_f.sum().clamp_min(1.0)
    return (kd * valid_f).sum() / denom * (temp * temp)


@dataclass
class EvalResult:
    loss: float
    miou: float
    per_class_iou: list[float]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: Dict[str, Any]) -> EvalResult:
    model.eval()
    num_classes = int(cfg["dataset"]["num_classes"])
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    ignore_indices = cfg["dataset"].get("metric_ignore_indices", [])
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    total = 0
    confmat = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            target = batch["mask"].to(device)
            logits = model(rgb, depth)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * rgb.size(0)
            total += rgb.size(0)
            pred = logits.argmax(1).detach().cpu()
            confmat += _confusion_matrix(pred, target.detach().cpu(), num_classes, ignore_index)
    miou, per_class = _miou(confmat, ignore_indices)
    return EvalResult(loss=total_loss / max(total, 1), miou=miou, per_class_iou=per_class)


def _benchmark(model: nn.Module, loader: DataLoader, device: torch.device, warmup: int, steps: int) -> Dict[str, float]:
    model.eval()
    batch = next(iter(loader))
    rgb = batch["rgb"].to(device)
    depth = batch["depth"].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(rgb, depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(rgb, depth)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency_ms = elapsed / max(steps, 1) * 1000.0
    return {"latency_ms": latency_ms, "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0}


def run(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    config_dir = config_path.resolve().parent
    dataset_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    dataset_root = _resolve_path(dataset_cfg["root"], config_dir)
    split_path = _resolve_path(dataset_cfg["split_path"], config_dir)
    geom_manifest = dataset_cfg.get("geometry_manifest")
    geom_manifest_path = _resolve_path(geom_manifest, config_dir) if geom_manifest else None
    distill_cfg = train_cfg.get("distillation", {})
    teacher_manifest = distill_cfg.get("teacher_manifest") if isinstance(distill_cfg, dict) else None
    teacher_manifest_path = _resolve_path(teacher_manifest, config_dir) if teacher_manifest else None
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    image_size = tuple(int(x) for x in dataset_cfg.get("image_size", [240, 320]))
    device = torch.device(train_cfg.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    seed = int(train_cfg.get("seed", 1337))
    torch.manual_seed(seed)
    np.random.seed(seed)

    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        ds = StudentUMDDataset(
            dataset_root=dataset_root,
            records=splits[split],
            geom_manifest=geom_manifest_path,
            teacher_manifest=teacher_manifest_path,
            image_size=(image_size[0], image_size[1]),
            ignore_index=int(dataset_cfg.get("ignore_index", 255)),
        )
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=(split == "train"),
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )

    class_weights = None
    loss_cfg = train_cfg.get("loss", {})
    if isinstance(loss_cfg, dict) and bool(loss_cfg.get("class_balance", False)):
        class_weights = _estimate_class_weights(
            dataset_root=dataset_root,
            records=splits["train"],
            num_classes=int(dataset_cfg["num_classes"]),
            ignore_index=int(dataset_cfg.get("ignore_index", 255)),
            ignore_indices=dataset_cfg.get("metric_ignore_indices", []),
            power=float(loss_cfg.get("class_balance_power", 0.5)),
            max_weight=float(loss_cfg.get("max_class_weight", 8.0)),
        )
        if "background_weight" in loss_cfg and class_weights.numel() > 0:
            class_weights[0] = float(loss_cfg["background_weight"])
        class_weights = class_weights.to(device)

    model = ReliabilityGatedStudent(
        num_classes=int(dataset_cfg["num_classes"]),
        width=int(model_cfg.get("width", 24)),
        fusion_mode=str(model_cfg.get("fusion_mode", "gate")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=int(dataset_cfg.get("ignore_index", 255)),
    )
    distill_enabled = isinstance(distill_cfg, dict) and bool(distill_cfg.get("enabled", False))
    distill_weight = float(distill_cfg.get("weight", 0.0)) if isinstance(distill_cfg, dict) else 0.0
    distill_temperature = float(distill_cfg.get("temperature", 2.0)) if isinstance(distill_cfg, dict) else 2.0
    distill_foreground_only = bool(distill_cfg.get("foreground_only", True)) if isinstance(distill_cfg, dict) else True
    max_epochs = int(train_cfg.get("max_epochs", 3))
    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        hard_loss_sum = 0.0
        distill_loss_sum = 0.0
        count = 0
        for batch in loaders["train"]:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            target = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(rgb, depth)
            hard_loss = criterion(logits, target)
            distill_loss = logits.new_tensor(0.0)
            if distill_enabled and distill_weight > 0.0 and "teacher_logits" in batch:
                distill_loss = _distillation_loss(
                    logits,
                    batch["teacher_logits"].to(device),
                    batch["teacher_available"].to(device),
                    target,
                    temperature=distill_temperature,
                    foreground_only=distill_foreground_only,
                    ignore_index=int(dataset_cfg.get("ignore_index", 255)),
                )
            loss = hard_loss + distill_weight * distill_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * rgb.size(0)
            hard_loss_sum += float(hard_loss.item()) * rgb.size(0)
            distill_loss_sum += float(distill_loss.item()) * rgb.size(0)
            count += rgb.size(0)
        val = _evaluate(model, loaders["val"], device, cfg)
        train_loss = loss_sum / max(count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_hard_loss": hard_loss_sum / max(count, 1),
                "train_distill_loss": distill_loss_sum / max(count, 1),
                "val_loss": val.loss,
                "val_miou": val.miou,
            }
        )
        if val.miou > best_val:
            best_val = val.miou
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, loaders["val"], device, cfg)
    test = _evaluate(model, loaders["test"], device, cfg)
    bench = _benchmark(
        model,
        loaders["test"],
        device,
        warmup=int(train_cfg.get("benchmark_warmup", 5)),
        steps=int(train_cfg.get("benchmark_steps", 20)),
    )
    params = sum(p.numel() for p in model.parameters())

    output_root = _resolve_path(train_cfg.get("output_root", "outputs/lightweight_student"), config_dir)
    run_dir = output_root / f"lightweight_student_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save({"state_dict": best_state, "config": cfg}, run_dir / "student.pth")
    summary = {
        "evidence_tier": str(
            train_cfg.get(
                "evidence_tier",
                "smoke" if bool(train_cfg.get("smoke", False)) else "capped_subset",
            )
        ),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "fusion_mode": str(model_cfg.get("fusion_mode", "gate")),
        "parameters": params,
        "depth_coverage": {
            split: {
                "available": int(ds.depth_available_count),
                "missing": int(ds.depth_missing_count),
                "total": int(len(ds)),
            }
            for split, ds in datasets.items()
        },
        "teacher_coverage": {
            split: {
                "available": int(ds.teacher_available_count),
                "missing": int(ds.teacher_missing_count),
                "total": int(len(ds)),
            }
            for split, ds in datasets.items()
        },
        "distillation": {
            "enabled": bool(distill_enabled),
            "weight": float(distill_weight),
            "temperature": float(distill_temperature),
            "foreground_only": bool(distill_foreground_only),
            "teacher_manifest": str(teacher_manifest_path) if teacher_manifest_path else None,
        },
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "val": val.__dict__,
        "test": test.__dict__,
        "benchmark": bench,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a lightweight RGB-D affordance student.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
