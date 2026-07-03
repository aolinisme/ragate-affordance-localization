import json
from pathlib import Path

import yaml

from pba.geometry.head_stats import compute_head_stats, run_head_stats


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_compute_head_stats_reports_expected_gate_overhead() -> None:
    config = {
        "dataset": {"num_classes": 8},
        "model": {
            "params": {
                "model_id": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
                "head": {
                    "feature_keys": [2, 5, 8, 11],
                    "use_batchnorm": True,
                    "use_geometry_gate": True,
                    "geometry_key": "geom_depth",
                },
            }
        },
    }

    stats = compute_head_stats(config)

    assert stats["fused_channels"] == 3076
    assert stats["total_parameters"] == 36920
    assert stats["baseline_head_parameters"] == 30768
    assert stats["extra_gate_parameters"] == 6152
    assert round(stats["gate_parameter_ratio_vs_head"], 4) == round(6152 / 30768, 4)


def test_run_head_stats_reads_config_and_prints_json(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "clip_gate.yaml"
    write_yaml(
        config_path,
        {
            "dataset": {"root": ".", "split_path": "split.json", "num_classes": 8},
            "model": {
                "target": "open_clip",
                "params": {
                    "model_id": "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
                    "head": {
                        "feature_keys": [2, 5, 8, 11],
                        "use_batchnorm": True,
                        "use_geometry_gate": True,
                        "geometry_key": "geom_depth",
                    },
                },
            },
            "training": {"output_root": "outputs"},
        },
    )

    stats = run_head_stats(["--config", str(config_path)])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)

    assert parsed["extra_gate_parameters"] == 6152
    assert stats["total_parameters"] == parsed["total_parameters"]
