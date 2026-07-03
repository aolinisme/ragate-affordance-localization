"""Attention map accumulation and export for interaction probing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

__all__ = ["AttentionAccumulator"]


class AttentionAccumulator:
    def __init__(self, token_map: Dict[str, int]):
        self.token_map = token_map
        self.storage: Dict[str, List[np.ndarray]] = {name: [] for name in token_map}

    def _infer_hw(self, length: int) -> int | None:
        side = int(round(math.sqrt(length)))
        if side * side == length:
            return side
        return None

    def add_from_probs(self, probs, token_dim: int) -> None:
        """Add torch attention probabilities shaped as (B, H, N_query, N_key)."""

        import torch

        with torch.no_grad():
            data = probs.detach().float().mean(dim=1)
            data = data[0]
            hw = self._infer_hw(data.shape[0])
            if hw is None:
                return
            attn_grid = data.view(hw, hw, token_dim)
            for name, index in self.token_map.items():
                if index >= token_dim:
                    continue
                plane = attn_grid[..., index].cpu().numpy()
                self.storage[name].append(plane)

    def summary(self) -> Dict[str, np.ndarray]:
        maps = {}
        for name, planes in self.storage.items():
            if not planes:
                continue
            stack = np.stack(planes, axis=0)
            maps[name] = stack.mean(axis=0)
        return maps

    def export(self, maps: Dict[str, np.ndarray], base_image: Image.Image, out_dir: Path) -> None:
        import matplotlib.cm as cm

        out_dir.mkdir(parents=True, exist_ok=True)
        base_rgb = base_image.convert("RGB")
        base_np = np.asarray(base_rgb, dtype=np.float32) / 255.0

        for name, arr in maps.items():
            if arr.size == 0:
                continue
            arr = arr - arr.min()
            denom = arr.max()
            norm = arr / denom if denom > 1e-8 else arr
            heat_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L").resize(
                base_rgb.size, Image.BICUBIC
            )
            heat = np.asarray(heat_img, dtype=np.float32) / 255.0
            colored = cm.get_cmap("viridis")(heat)[..., :3]
            overlay = (0.65 * colored + 0.35 * base_np).clip(0.0, 1.0)

            Image.fromarray((colored * 255).astype(np.uint8)).save(out_dir / f"{name}_heat.png")
            Image.fromarray((overlay * 255).astype(np.uint8)).save(out_dir / f"{name}_overlay.png")
            np.save(out_dir / f"{name}_heat.npy", heat)
