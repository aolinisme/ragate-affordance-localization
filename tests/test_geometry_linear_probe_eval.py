from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml

from pba.geometry.linear_probe import parse_eval_args, run_eval


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHead:
    def __init__(self) -> None:
        self.loaded_state = None

    def to(self, device: str):
        self.device = device
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        self.loaded_state = state_dict


class FakeExperiment:
    instances = []

    def __init__(self, config) -> None:
        self.config = config
        self.device = "cpu"
        self.training_cfg = {"precision": "fp32"}
        self.num_classes = 8
        self.ignore_index = 255
        self.cfg = {"visualization": {"num_samples": 2}}
        self.val_loader = "val_loader"
        self.test_loader = "test_loader"
        self.val_log_interval = 5
        self.metric_ignore_indices = [0]
        self.use_multi_head = False
        self.backbone = "backbone"
        self.target_layer = 11
        self.head = FakeHead()
        FakeExperiment.instances.append(self)

    def _build_head(self) -> FakeHead:
        return self.head


class FakeTorch:
    saved = []

    class nn:
        class CrossEntropyLoss:
            def __init__(self, ignore_index: int) -> None:
                self.ignore_index = ignore_index

    @staticmethod
    def load(path: Path, map_location: str):
        return {"state_dict": {"weight": "loaded"}}

    @staticmethod
    def save(data, path: Path) -> None:
        FakeTorch.saved.append((data, path))


def fake_logger_factory(*args, **kwargs):
    class Logger:
        def info(self, *args, **kwargs) -> None:
            pass

    return Logger()


def fake_evaluate_fn(*args, **kwargs):
    fake_evaluate_fn.calls.append({"args": args, "kwargs": kwargs})
    return {"mIoU": 0.5}, [{"example": 1}]


fake_evaluate_fn.calls = []


def test_parse_eval_args_accepts_checkpoint_config_and_split() -> None:
    args = parse_eval_args(
        [
            "linear_probe.pth",
            "--config",
            "configs/dinov2.yaml",
            "--split",
            "val",
            "--num-examples",
            "3",
        ]
    )

    assert args.checkpoint == Path("linear_probe.pth")
    assert args.config == Path("configs/dinov2.yaml")
    assert args.split == "val"
    assert args.num_examples == 3


def test_run_eval_writes_metrics_and_optional_examples(tmp_path: Path) -> None:
    FakeExperiment.instances.clear()
    FakeTorch.saved.clear()
    fake_evaluate_fn.calls.clear()
    config_path = tmp_path / "umd_linear_probing" / "configs" / "dinov2.yaml"
    checkpoint = tmp_path / "linear_probe.pth"
    write_yaml(
        config_path,
        {
            "dataset": {"root": "../../datasets/UMD"},
            "model": {"target": "dinov2", "params": {"name": "dinov2"}},
            "training": {"output_root": "outputs"},
        },
    )

    metrics, examples = run_eval(
        [str(checkpoint), "--config", str(config_path), "--split", "test", "--save-examples"],
        experiment_cls=FakeExperiment,
        evaluate_fn=fake_evaluate_fn,
        logger_factory=fake_logger_factory,
        torch_module=FakeTorch,
    )

    experiment = FakeExperiment.instances[0]
    assert experiment.head.loaded_state == {"weight": "loaded"}
    assert fake_evaluate_fn.calls[0]["args"][2] == "test_loader"
    assert fake_evaluate_fn.calls[0]["kwargs"]["split"] == "test"
    assert fake_evaluate_fn.calls[0]["kwargs"]["max_examples"] == 2
    assert metrics == {"mIoU": 0.5}
    assert examples == [{"example": 1}]
    assert checkpoint.with_suffix(".test_metrics.json").read_text(encoding="utf-8") == '{\n  "mIoU": 0.5\n}'
    assert FakeTorch.saved == [([{"example": 1}], checkpoint.with_suffix(".examples.pt"))]


def test_run_eval_num_examples_overrides_config(tmp_path: Path) -> None:
    FakeExperiment.instances.clear()
    FakeTorch.saved.clear()
    fake_evaluate_fn.calls.clear()
    config_path = tmp_path / "umd_linear_probing" / "configs" / "dinov2.yaml"
    checkpoint = tmp_path / "linear_probe.pth"
    write_yaml(
        config_path,
        {
            "dataset": {"root": "../../datasets/UMD"},
            "model": {"target": "dinov2", "params": {"name": "dinov2"}},
            "training": {"output_root": "outputs"},
        },
    )

    run_eval(
        [
            str(checkpoint),
            "--config",
            str(config_path),
            "--split",
            "test",
            "--num-examples",
            "4",
        ],
        experiment_cls=FakeExperiment,
        evaluate_fn=fake_evaluate_fn,
        logger_factory=fake_logger_factory,
        torch_module=FakeTorch,
    )

    assert fake_evaluate_fn.calls[0]["kwargs"]["max_examples"] == 4


def test_legacy_eval_script_reexports_package_entrypoint() -> None:
    legacy = load_module(
        REPO_ROOT / "geometry_probing/umd_linear_probing/scripts/eval.py",
        "legacy_geometry_eval_script",
    )

    assert legacy.parse_eval_args is parse_eval_args
    assert legacy.run_eval is run_eval
