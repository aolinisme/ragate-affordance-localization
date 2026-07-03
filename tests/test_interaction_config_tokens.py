from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

from pba.interaction.config import parse_probe_args
from pba.interaction.tokens import collect_token_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEncoded:
    def __init__(self, ids):
        self.input_ids = [ids]


class FakeTokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, prompt: str, return_tensors: str, add_special_tokens: bool):
        assert prompt == "hold the mug"
        assert return_tensors == "pt"
        assert add_special_tokens is True
        return FakeEncoded([0, 1, 2, 3])

    def convert_ids_to_tokens(self, ids):
        return self.tokens


def test_parse_probe_args_preserves_existing_defaults() -> None:
    args = parse_probe_args(
        [
            "--model-id",
            "models/FLUX.1-Kontext-dev",
            "--image",
            "input.png",
            "--prompt",
            "hold the mug",
            "--affordance",
            "hold",
        ]
    )

    assert args.model_id == "models/FLUX.1-Kontext-dev"
    assert args.image == Path("input.png")
    assert args.prompt == "hold the mug"
    assert args.affordance == "hold"
    assert args.output_root == Path("probe_outputs")
    assert args.steps == 20
    assert args.guidance == 3.0
    assert args.seed == 0
    assert args.device == "cuda"


def test_collect_token_index_matches_exact_or_subtoken() -> None:
    tokenizer = FakeTokenizer(["</s>", "▁object", "▁holding", "▁mug"])

    assert collect_token_index(tokenizer, "hold the mug", "hold") == {"hold": 2}


def test_collect_token_index_returns_empty_when_missing() -> None:
    tokenizer = FakeTokenizer(["</s>", "▁object", "▁mug"])

    assert collect_token_index(tokenizer, "hold the mug", "press") == {}


def test_legacy_interaction_script_uses_package_token_helper() -> None:
    legacy = load_module(
        REPO_ROOT / "interaction_probing/cross_attention_probe/cross_attention_probe.py",
        "legacy_interaction_probe",
    )
    tokenizer = FakeTokenizer(["</s>", "▁holding", "▁mug"])

    assert legacy._collect_token_index(tokenizer, "hold the mug", "hold") == {"hold": 1}
