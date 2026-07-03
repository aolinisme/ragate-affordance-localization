from pba.fusion.prompts import (
    build_object_token_candidates,
    format_object_name,
    sanitize_token_name,
    select_token,
)


def test_format_object_name_replaces_underscores() -> None:
    assert format_object_name("coffee_mug") == "coffee mug"


def test_select_token_prefers_first_candidate_match() -> None:
    tokens = [
        {"id": 0, "tok": "</s>"},
        {"id": 1, "tok": "▁hold"},
        {"id": 2, "tok": "▁mug"},
    ]

    assert select_token(tokens, ["mug", "hold"]) == {"index": 2, "token": "▁mug"}


def test_select_token_falls_back_to_first_non_special_token() -> None:
    tokens = [
        {"id": 0, "tok": "</s>"},
        {"id": 1, "tok": "▁"},
        {"id": 2, "tok": "object"},
    ]

    assert select_token(tokens, ["missing"]) == {"index": 2, "token": "object"}


def test_select_token_returns_none_when_no_usable_token() -> None:
    assert select_token([{"id": 0, "tok": "</s>"}, {"id": 1, "tok": "▁"}], ["missing"]) is None


def test_select_token_can_disable_fallback() -> None:
    tokens = [{"id": 0, "tok": "▁hold"}, {"id": 1, "tok": "▁cup"}]

    assert select_token(tokens, ["missing"], allow_fallback=False) is None


def test_sanitize_token_name_replaces_path_separators_and_truncates() -> None:
    assert sanitize_token_name("cup/with\\handle token") == "cup_with_handle_token"
    assert sanitize_token_name("abcdefghijklmnopqrstuvwxyz") == "abcdefghijklmnopqrstuvwx"


def test_build_object_token_candidates_matches_existing_variants() -> None:
    candidates = build_object_token_candidates("coffee-mug_big")

    assert "coffee mug big" in candidates
    assert "▁coffee mug big" in candidates
    assert "coffeemugbig" in candidates
    assert "▁Big" in candidates
    assert len(candidates) == len(set(candidates))


def test_build_object_token_candidates_includes_simple_singulars() -> None:
    candidates = build_object_token_candidates("skis")

    assert "ski" in candidates
    assert "▁ski" in candidates


def test_select_token_uses_deterministic_compound_candidate_order() -> None:
    tokens = [
        {"id": 0, "tok": "▁hold"},
        {"id": 1, "tok": "▁golf"},
        {"id": 2, "tok": "▁clubs"},
    ]
    candidates = build_object_token_candidates("golf_clubs")

    assert select_token(tokens, candidates, allow_fallback=False) == {"index": 1, "token": "▁golf"}
