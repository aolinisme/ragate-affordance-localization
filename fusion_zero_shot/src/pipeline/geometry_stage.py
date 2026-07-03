from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Sequence


def smooth_map(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return arr
    return cv2.GaussianBlur(arr, (0, 0), sigma)


def largest_component(base_mask: np.ndarray, priority_mask: Optional[np.ndarray] = None) -> np.ndarray:
    base_u8 = (base_mask > 0).astype(np.uint8)
    if base_u8.sum() == 0:
        return base_u8.astype(np.float32)
    num, labels = cv2.connectedComponents(base_u8)
    if num <= 1:
        return base_u8.astype(np.float32)

    best_label = 1
    if priority_mask is not None and priority_mask.sum() > 0:
        scores = []
        for lab in range(1, num):
            overlap = int(((labels == lab) & (priority_mask > 0)).sum())
            scores.append((overlap, np.count_nonzero(labels == lab), lab))
        scores.sort(reverse=True)
        best_label = scores[0][2] if scores else 1
    else:
        areas = [(np.count_nonzero(labels == lab), lab) for lab in range(1, num)]
        areas.sort(reverse=True)
        best_label = areas[0][1]

    return (labels == best_label).astype(np.float32)


def _normalize_for_cosine(arr: np.ndarray) -> Optional[np.ndarray]:
    vec = np.asarray(arr, dtype=np.float32).reshape(-1)
    if vec.size == 0:
        return None
    vmin = float(vec.min())
    vmax = float(vec.max())
    if abs(vmax - vmin) < 1e-6:
        return None
    vec = (vec - vmin) / (vmax - vmin)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _softmax_prob(
    arr: np.ndarray,
    *,
    temperature: float,
    dirichlet: float,
    use_log1p: bool,
    eps: float,
) -> np.ndarray:
    """Convert a heatmap into a probability map via temperature-scaled softmax."""

    clamped = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    if use_log1p:
        clamped = np.log1p(clamped)

    flat = clamped.reshape(-1)
    if flat.size == 0:
        return clamped

    temp = float(max(temperature, eps))
    scaled = flat / temp
    scaled -= float(np.max(scaled))
    exp_vals = np.exp(scaled)
    denom = float(exp_vals.sum()) + eps
    prob = exp_vals / denom

    smooth = float(max(dirichlet, 0.0))
    if smooth > 0.0:
        smooth = min(smooth, 1.0)
        uniform = np.full_like(prob, 1.0 / prob.size, dtype=np.float32)
        prob = prob * (1.0 - smooth) + uniform * smooth
        prob = prob / (float(prob.sum()) + eps)

    return prob.reshape(arr.shape)


def _soft_fuse_heatmaps(
    verb_map: np.ndarray,
    geom_energy: np.ndarray,
    *,
    lam: float,
    gamma: float,
    temperature: float,
    dirichlet: float,
    use_log1p: bool,
    eps: float = 1e-6,
) -> np.ndarray:
    lam = float(np.clip(lam, 0.0, 1.0))
    gamma = float(max(gamma, eps))

    verb_prob = _softmax_prob(
        verb_map,
        temperature=temperature,
        dirichlet=dirichlet,
        use_log1p=use_log1p,
        eps=eps,
    )
    geom_pre = np.power(np.clip(geom_energy, 0.0, None), gamma)
    geom_prob = _softmax_prob(
        geom_pre,
        temperature=temperature,
        dirichlet=dirichlet,
        use_log1p=use_log1p,
        eps=eps,
    )

    mix_log = lam * np.log(verb_prob + eps) + (1.0 - lam) * np.log(geom_prob + eps)
    temp = float(max(temperature, eps))
    mix_log = mix_log / temp
    fused = np.exp(mix_log)
    fused = np.clip(fused, 0.0, None)
    total = float(fused.sum())
    if total <= eps:
        return verb_prob.astype(np.float32)
    return (fused / total).astype(np.float32)


def _build_topk_mask(
    verb_map: np.ndarray,
    percent: float,
) -> Optional[np.ndarray]:
    arr = np.clip(np.asarray(verb_map, dtype=np.float32), 0.0, None)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return None
    pct = float(np.clip(percent, 0.1, 100.0))
    cutoff = np.percentile(flat, max(0.0, 100.0 - pct))
    mask = (arr >= cutoff).astype(np.uint8)
    if mask.sum() == 0:
        idx = int(np.argmax(flat))
        mask.reshape(-1)[idx] = 1
    return mask


def _nss_score(channel: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> Optional[float]:
    arr = np.asarray(channel, dtype=np.float32)
    valid = mask > 0
    count = int(valid.sum())
    if count == 0:
        return None
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        return None
    z = (arr - mean) / std
    return float(z[valid].mean())


def _topk_energy_diff(channel: np.ndarray, mask: np.ndarray) -> Optional[float]:
    arr = np.asarray(channel, dtype=np.float32)
    valid = mask > 0
    count = int(valid.sum())
    if count == 0:
        return None
    fg = arr[valid]
    bg = arr[~valid]
    if fg.size == 0:
        return None
    fg_mean = float(fg.mean())
    bg_mean = float(bg.mean()) if bg.size > 0 else 0.0
    return fg_mean - bg_mean


def _rank_topk_mask(arr: np.ndarray, percent: float) -> Optional[np.ndarray]:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    flat = values.reshape(-1)
    if flat.size == 0:
        return None

    pct = float(np.clip(percent, 0.1, 100.0))
    k = max(1, int(np.ceil(flat.size * pct / 100.0)))
    if k >= flat.size:
        indices = np.arange(flat.size)
    else:
        indices = np.argpartition(flat, -k)[-k:]

    mask = np.zeros(flat.size, dtype=np.uint8)
    mask[indices] = 1
    return mask.reshape(values.shape)


def _heatmap_confidence(
    arr: np.ndarray,
    *,
    topk_percent: float = 5.0,
    eps: float = 1e-6,
) -> Dict[str, Optional[float]]:
    values = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    flat = values.reshape(-1)
    if flat.size == 0:
        return {"confidence": 0.0, "nss": None, "topk_diff": None}

    vmin = float(flat.min())
    vmax = float(flat.max())
    if abs(vmax - vmin) < eps:
        return {"confidence": 0.0, "nss": None, "topk_diff": None}

    mask = _rank_topk_mask(values, topk_percent)
    if mask is None:
        return {"confidence": 0.0, "nss": None, "topk_diff": None}

    nss = _nss_score(values, mask, eps=eps)
    topk_diff = _topk_energy_diff(values, mask)
    nss_conf = float(np.clip(np.tanh(max(nss or 0.0, 0.0) / 2.0), 0.0, 1.0))
    diff_scale = max(vmax - vmin, eps)
    diff_conf = float(np.clip(max(topk_diff or 0.0, 0.0) / diff_scale, 0.0, 1.0))
    confidence = float(np.clip(0.5 * nss_conf + 0.5 * diff_conf, 0.0, 1.0))
    return {
        "confidence": confidence,
        "nss": float(nss) if nss is not None else None,
        "topk_diff": float(topk_diff) if topk_diff is not None else None,
    }


def _cosine_alignment_confidence(
    first: np.ndarray,
    second: np.ndarray,
) -> Optional[float]:
    first_vec = _normalize_for_cosine(first)
    second_vec = _normalize_for_cosine(second)
    if first_vec is None or second_vec is None:
        return None
    cosine = float(np.dot(first_vec, second_vec))
    return float(np.clip(cosine, 0.0, 1.0))


def _compute_adaptive_fusion_meta(
    verb_map: np.ndarray,
    geom_energy: np.ndarray,
    *,
    base_lambda: float,
    min_lambda: float,
    max_lambda: float,
    verb_weight: float,
    geometry_weight: float,
    alignment_weight: float,
    eps: float = 1e-6,
) -> Dict[str, object]:
    base = float(np.clip(base_lambda, 0.0, 1.0))
    lo = float(np.clip(min_lambda, 0.0, 1.0))
    hi = float(np.clip(max_lambda, 0.0, 1.0))
    if lo > hi:
        lo, hi = hi, lo

    v_weight = float(max(verb_weight, 0.0))
    g_weight = float(max(geometry_weight, 0.0))
    a_weight = float(max(alignment_weight, 0.0))

    interaction = _heatmap_confidence(verb_map)
    geometry = _heatmap_confidence(geom_energy)
    alignment_val = _cosine_alignment_confidence(verb_map, geom_energy)
    alignment_conf = float(alignment_val if alignment_val is not None else 0.0)

    interaction_conf = float(interaction["confidence"] or 0.0)
    geometry_conf = float(geometry["confidence"] or 0.0)
    interaction_evidence = v_weight * interaction_conf
    geometry_evidence = g_weight * geometry_conf + a_weight * alignment_conf
    total_evidence = interaction_evidence + geometry_evidence

    if total_evidence <= eps:
        adaptive_lambda = base
    else:
        interaction_ratio = interaction_evidence / total_evidence
        adaptive_lambda = lo + (hi - lo) * interaction_ratio
    adaptive_lambda = float(np.clip(adaptive_lambda, lo, hi))

    return {
        "adaptive_fusion_enabled": True,
        "adaptive_lambda": adaptive_lambda,
        "interaction_confidence": interaction_conf,
        "geometry_confidence": geometry_conf,
        "alignment_confidence": alignment_conf,
        "base_lambda": base,
        "lambda_range": float(hi - lo),
        "min_lambda": lo,
        "max_lambda": hi,
        "interaction_evidence": float(interaction_evidence),
        "geometry_evidence": float(geometry_evidence),
        "interaction_nss": interaction["nss"],
        "interaction_topk_diff": interaction["topk_diff"],
        "geometry_nss": geometry["nss"],
        "geometry_topk_diff": geometry["topk_diff"],
    }


def _attention_priority_choice(
    stack: np.ndarray,
    verb_map: np.ndarray,
    topk_percent: float,
    nss_weight: float,
    topk_weight: float,
) -> Optional[Dict[str, object]]:
    if stack.ndim != 3 or stack.shape[2] == 0:
        return None
    if nss_weight == 0.0 and topk_weight == 0.0:
        return None

    mask = _build_topk_mask(verb_map, topk_percent)
    if mask is None:
        return None

    scores: list[Optional[float]] = []
    for idx in range(stack.shape[2]):
        ch = stack[..., idx]
        score_val = 0.0
        valid = False
        if nss_weight != 0.0:
            nss = _nss_score(ch, mask)
            if nss is not None:
                score_val += nss_weight * nss
                valid = True
        if topk_weight != 0.0:
            diff = _topk_energy_diff(ch, mask)
            if diff is not None:
                score_val += topk_weight * diff
                valid = True
        scores.append(score_val if valid else None)

    valid_pairs = [(idx, sc) for idx, sc in enumerate(scores) if sc is not None]
    if not valid_pairs:
        return None
    best_idx = max(valid_pairs, key=lambda x: x[1])[0]
    return {
        "index": best_idx,
        "scores": scores,
        "topk_pixels": int(mask.sum()),
    }


def generate_geometry_mask(
    pcs_orig: np.ndarray,
    *,
    smooth_sigma: float = 1.2,
    binary_threshold: float = 0.5,
    verb_map: Optional[np.ndarray] = None,
    enable_soft_fusion: bool = False,
    soft_lambda: float = 0.65,
    soft_gamma: float = 0.7,
    soft_temperature: float | Sequence[float] = 1.15,
    soft_dirichlet: float = 0.008,
    soft_use_log1p: bool = False,
    max_channels: Optional[int] = None,
    forced_pc_index: Optional[int] = None,
    use_attention_fallback: bool = False,
    attention_topk_percent: float = 10.0,
    attention_nss_weight: float = 1.0,
    attention_topk_weight: float = 1.0,
    adaptive_fusion: bool = False,
    gate_min_lambda: float = 0.35,
    gate_max_lambda: float = 0.85,
    gate_verb_weight: float = 0.45,
    gate_geometry_weight: float = 0.35,
    gate_alignment_weight: float = 0.20,
    gate_similarity_floor: Optional[float] = None,
    gate_fallback_lambda: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    if pcs_orig.ndim != 3 or pcs_orig.shape[2] == 0:
        raise ValueError("pcs_orig must be an HxWxC array with at least one channel")

    total_channels = pcs_orig.shape[2]
    if max_channels is None:
        num_channels = min(3, total_channels)
    else:
        num_channels = max(1, min(int(max_channels), total_channels))

    channels = []
    for ch in range(num_channels):
        channels.append(smooth_map(np.clip(pcs_orig[..., ch], 0.0, 1.0), smooth_sigma))
    stack = np.stack(channels, axis=2)

    weights = np.ones(num_channels, dtype=np.float32) / max(1, num_channels)
    pc_labels: Sequence[str] = [f"pc{idx + 1}" for idx in range(num_channels)]
    similarity_scores: Optional[list[float]] = None
    similarity_method: Optional[str] = None
    attention_meta: Optional[Dict[str, object]] = None
    selected_idx = 0

    forced_idx = None
    if forced_pc_index is not None:
        forced_idx = int(forced_pc_index) - 1
        if forced_idx < 0 or forced_idx >= num_channels:
            raise ValueError(
                f"forced_pc_index {forced_pc_index} out of range (1~{num_channels})"
            )

    if forced_idx is not None:
        selected_idx = forced_idx
        energy = np.clip(stack[..., selected_idx], 0.0, None)
        weights = np.zeros_like(weights)
        weights[selected_idx] = 1.0
    else:
        attention_choice = None
        if use_attention_fallback and verb_map is not None:
            attention_choice = _attention_priority_choice(
                stack,
                verb_map,
                attention_topk_percent,
                attention_nss_weight,
                attention_topk_weight,
            )
        if attention_choice is not None:
            selected_idx = int(attention_choice["index"])
            similarity_scores = attention_choice["scores"]
            similarity_method = "attention"
            energy = np.clip(stack[..., selected_idx], 0.0, None)
            weights = np.zeros_like(weights)
            weights[selected_idx] = 1.0
            attention_meta = {
                "topk_percent": float(np.clip(attention_topk_percent, 0.1, 100.0)),
                "topk_pixels": int(attention_choice.get("topk_pixels", 0)),
                "nss_weight": float(attention_nss_weight),
                "topk_weight": float(attention_topk_weight),
            }
        else:
            if verb_map is not None:
                verb_vec = _normalize_for_cosine(verb_map)
                if verb_vec is not None:
                    similarity_scores = []
                    valid_pairs = []
                    for idx in range(num_channels):
                        channel_vec = _normalize_for_cosine(stack[..., idx])
                        if channel_vec is None:
                            similarity_scores.append(None)
                            continue
                        cosine = float(np.dot(channel_vec, verb_vec))
                        similarity_scores.append(cosine)
                        valid_pairs.append((idx, cosine))
                    if valid_pairs:
                        selected_idx = max(valid_pairs, key=lambda x: x[1])[0]
                        similarity_method = "cosine"
                    else:
                        similarity_scores = None

            if similarity_scores is not None:
                energy = np.clip(stack[..., selected_idx], 0.0, None)
                weights = np.zeros_like(weights)
                weights[selected_idx] = 1.0
            else:
                energy = np.clip(stack[..., 0], 0.0, None)
                selected_idx = 0
                similarity_scores = similarity_scores or []
                similarity_method = None

    denom = energy.max()
    if denom < 1e-6:
        energy_norm = np.zeros_like(energy, dtype=np.float32)
    else:
        energy_norm = energy / denom
    energy_norm = energy_norm.astype(np.float32)

    geom_mask = (energy_norm >= binary_threshold).astype(np.float32)

    if isinstance(soft_temperature, (list, tuple)):
        temp_list = [float(t) for t in soft_temperature if t is not None]
    else:
        temp_list = [float(soft_temperature)]
    temp_list = [t if t > 1e-6 else 1e-6 for t in temp_list]
    if not temp_list:
        temp_list = [1.15]

    soft_fusion = None
    soft_fusion_multi: list[dict[str, object]] = []
    adaptive_meta: Optional[Dict[str, object]] = None
    fusion_lambda = float(soft_lambda)
    if enable_soft_fusion and verb_map is not None:
        if adaptive_fusion:
            selected_similarity = None
            if similarity_scores is not None and 0 <= selected_idx < len(similarity_scores):
                selected_similarity = similarity_scores[selected_idx]
            adaptive_meta = _compute_adaptive_fusion_meta(
                verb_map,
                energy_norm,
                base_lambda=soft_lambda,
                min_lambda=gate_min_lambda,
                max_lambda=gate_max_lambda,
                verb_weight=gate_verb_weight,
                geometry_weight=gate_geometry_weight,
                alignment_weight=gate_alignment_weight,
            )
            adaptive_meta["similarity_floor"] = (
                float(gate_similarity_floor) if gate_similarity_floor is not None else None
            )
            adaptive_meta["selected_similarity"] = (
                float(selected_similarity) if selected_similarity is not None else None
            )
            adaptive_meta["fallback_lambda"] = (
                float(gate_fallback_lambda) if gate_fallback_lambda is not None else None
            )
            adaptive_meta["fallback_used"] = False
            if gate_similarity_floor is not None:
                floor = float(gate_similarity_floor)
                if selected_similarity is None or float(selected_similarity) < floor:
                    fallback = soft_lambda if gate_fallback_lambda is None else float(gate_fallback_lambda)
                    fallback = float(np.clip(fallback, 0.0, 1.0))
                    adaptive_meta["adaptive_lambda_before_fallback"] = adaptive_meta["adaptive_lambda"]
                    adaptive_meta["adaptive_lambda"] = fallback
                    adaptive_meta["fallback_used"] = True
            fusion_lambda = float(adaptive_meta["adaptive_lambda"])
        for temp in temp_list:
            fused = _soft_fuse_heatmaps(
                verb_map,
                energy_norm,
                lam=fusion_lambda,
                gamma=soft_gamma,
                temperature=temp,
                dirichlet=soft_dirichlet,
                use_log1p=soft_use_log1p,
            )
            entry = {
                "map": fused,
                "temperature": float(temp),
                "params": {
                    "lambda": fusion_lambda,
                    "base_lambda": soft_lambda,
                    "gamma": soft_gamma,
                    "temperature": float(temp),
                    "dirichlet": soft_dirichlet,
                    "log1p": bool(soft_use_log1p),
                    "adaptive_fusion": adaptive_meta,
                },
            }
            soft_fusion_multi.append(entry)
        if soft_fusion_multi:
            soft_fusion = soft_fusion_multi[0]

    result = {
        "energy": energy_norm,
        "mask": geom_mask,
        "weights": weights,
        "thresholds": {"binary": binary_threshold},
        "selected_pc": pc_labels[selected_idx],
        "pc_index": int(selected_idx),
        "similarity_scores": similarity_scores if similarity_scores is not None else None,
        "similarity_method": similarity_method,
        "attention_meta": attention_meta,
        "soft_fusion": soft_fusion,
        "soft_fusion_multi": soft_fusion_multi if soft_fusion_multi else None,
        "soft_temperatures": temp_list,
        "channels_used": num_channels,
        "pc_labels": list(pc_labels),
    }
    if adaptive_meta is not None:
        result["adaptive_fusion"] = adaptive_meta
    return result
