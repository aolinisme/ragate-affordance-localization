"""Geometry linear probing train/eval entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from pba.geometry.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PROJECT_ROOT = REPO_ROOT / "geometry_probing" / "umd_linear_probing"

ExperimentFactory = Callable[[Any], Any]

__all__ = [
    "build_eval_parser",
    "build_train_parser",
    "parse_eval_args",
    "parse_train_args",
    "run_eval",
    "run_train",
]


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the UMD linear probing experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Single configuration file containing the full experiment setup.",
    )
    parser.add_argument(
        "--defaults",
        type=Path,
        default=None,
        help="[Deprecated] Base config file to be merged with --local.",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="[Optional] Local override config, merged into --defaults.",
    )
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_train_parser().parse_args(argv)


def run_train(
    argv: Sequence[str] | None = None,
    *,
    experiment_cls: ExperimentFactory | None = None,
) -> Any:
    args = parse_train_args(argv)
    if args.config is None or not args.config.exists():
        raise FileNotFoundError("Please provide --config pointing to a valid YAML file.")

    config = load_config(args.config, args.local)
    if experiment_cls is None:
        experiment_cls = _load_legacy_experiment_cls()
    experiment = experiment_cls(config)
    experiment.train()
    return experiment


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved linear probe checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved checkpoint (.pth)")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Single configuration file containing the full experiment setup.",
    )
    parser.add_argument(
        "--local",
        type=Path,
        default=None,
        help="[Optional] Local override config, merged into --config.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--save-examples",
        action="store_true",
        help="If set, dumps qualitative examples next to the checkpoint.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of qualitative examples to collect during evaluation.",
    )
    return parser


def parse_eval_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_eval_parser().parse_args(argv)


def run_eval(
    argv: Sequence[str] | None = None,
    *,
    experiment_cls: ExperimentFactory | None = None,
    evaluate_fn: Callable[..., tuple[dict[str, Any], Any]] | None = None,
    logger_factory: Callable[..., Any] | None = None,
    torch_module: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    args = parse_eval_args(argv)
    if not args.config.exists():
        raise FileNotFoundError("Please provide --config pointing to a valid YAML file.")

    config = load_config(args.config, args.local)
    if experiment_cls is None:
        experiment_cls = _load_legacy_experiment_cls()
    if evaluate_fn is None:
        evaluate_fn = _load_legacy_evaluate_fn()
    if logger_factory is None:
        logger_factory = _load_legacy_logger_factory()
    if torch_module is None:
        import torch as torch_module

    experiment = experiment_cls(config)
    eval_logger = logger_factory(
        args.checkpoint.parent,
        name=f"linear_probe_eval.{args.split}",
        filename=f"eval_{args.split}.log",
    )

    head = experiment._build_head().to(experiment.device)
    checkpoint = torch_module.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    head.load_state_dict(state_dict)

    loader = experiment.val_loader if args.split == "val" else experiment.test_loader
    num_examples = args.num_examples
    if num_examples is None:
        num_examples = experiment.cfg.get("visualization", {}).get("num_samples", 0)

    metrics, examples = evaluate_fn(
        experiment.backbone,
        head,
        loader,
        experiment.device,
        precision=experiment.training_cfg.get("precision", "bf16"),
        num_classes=experiment.num_classes,
        ignore_index=experiment.ignore_index,
        criterion=torch_module.nn.CrossEntropyLoss(ignore_index=experiment.ignore_index),
        target_layer=experiment.target_layer,
        max_examples=num_examples,
        logger=eval_logger,
        log_interval=experiment.val_log_interval,
        ignore_indices=experiment.metric_ignore_indices,
        split=args.split,
        use_multi_head=experiment.use_multi_head,
    )

    eval_logger.info("%s metrics: %s", args.split, metrics)

    metrics_path = args.checkpoint.with_suffix(f".{args.split}_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))

    if args.save_examples and examples:
        out_path = args.checkpoint.with_suffix(".examples.pt")
        torch_module.save(examples, out_path)
        print(f"Saved qualitative examples to {out_path}")

    return metrics, examples


def _ensure_legacy_project_root() -> None:
    if str(LEGACY_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(LEGACY_PROJECT_ROOT))


def _load_legacy_experiment_cls() -> ExperimentFactory:
    _ensure_legacy_project_root()
    from src.engine.trainer import LinearProbeExperiment

    return LinearProbeExperiment


def _load_legacy_evaluate_fn() -> Callable[..., tuple[dict[str, Any], Any]]:
    _ensure_legacy_project_root()
    from src.engine.eval import evaluate_linear_probe

    return evaluate_linear_probe


def _load_legacy_logger_factory() -> Callable[..., Any]:
    _ensure_legacy_project_root()
    from src.utils.logging import create_logger

    return create_logger
