from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess


@dataclass(frozen=True)
class PublicReleaseFinding:
    level: str
    code: str
    path: str
    message: str


PRIVATE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(Users|home|mnt|data)/")

PUBLIC_PLACEHOLDER_PATTERNS = (
    "/path/to/",
)

TEXT_FILE_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_PATH_SKIP_PREFIXES = (
    "docs/",
    "tests/",
    "fusion_zero_shot/src/dino/third_party/",
)

PRIVATE_PATH_SKIP_SUFFIXES = (
    ".local.example.yaml",
    ".local.example.yml",
)

DEFAULT_MAX_TRACKED_FILE_BYTES = 50 * 1024 * 1024


def get_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_FILE_SUFFIXES


def _should_scan_private_paths(relative_path: str) -> bool:
    if relative_path.startswith(PRIVATE_PATH_SKIP_PREFIXES):
        return False
    return not relative_path.endswith(PRIVATE_PATH_SKIP_SUFFIXES)


def _is_comment_line(line: str) -> bool:
    return line.lstrip().startswith("#")


def _contains_public_placeholder(line: str) -> bool:
    if any(pattern in line for pattern in PUBLIC_PLACEHOLDER_PATTERNS):
        return True
    return "..." in line and PRIVATE_PATH_RE.search(line) is not None


def check_private_paths(repo_root: Path, *, tracked_files: list[str] | None = None) -> list[PublicReleaseFinding]:
    tracked = tracked_files if tracked_files is not None else get_tracked_files(repo_root)
    findings: list[PublicReleaseFinding] = []

    for relative_path in tracked:
        path = repo_root / relative_path
        if not _should_scan_private_paths(relative_path):
            continue
        if not path.exists() or not path.is_file() or not _is_text_candidate(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if _is_comment_line(line) or _contains_public_placeholder(line):
                continue
            match = PRIVATE_PATH_RE.search(line)
            if match:
                pattern = f"/{match.group(1)}/"
                findings.append(
                    PublicReleaseFinding(
                        level="BLOCKED",
                        code="private-path",
                        path=relative_path,
                        message=f"private local path pattern {pattern!r} found",
                    )
                )
            if findings and findings[-1].path == relative_path:
                break

    return findings


def check_gitignore_entry(repo_root: Path, pattern: str) -> list[PublicReleaseFinding]:
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return [
            PublicReleaseFinding(
                level="BLOCKED",
                code="missing-gitignore-entry",
                path=".gitignore",
                message=f"{pattern} is not ignored",
            )
        ]

    lines = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if pattern in lines:
        return []
    return [
        PublicReleaseFinding(
            level="BLOCKED",
            code="missing-gitignore-entry",
            path=".gitignore",
            message=f"{pattern} is not ignored",
        )
    ]


def check_tracked_file_sizes(
    repo_root: Path,
    *,
    tracked_files: list[str] | None = None,
    max_bytes: int = DEFAULT_MAX_TRACKED_FILE_BYTES,
) -> list[PublicReleaseFinding]:
    tracked = tracked_files if tracked_files is not None else get_tracked_files(repo_root)
    findings: list[PublicReleaseFinding] = []

    for relative_path in tracked:
        path = repo_root / relative_path
        if not path.exists() or not path.is_file():
            continue
        size = path.stat().st_size
        if size <= max_bytes:
            continue
        findings.append(
            PublicReleaseFinding(
                level="BLOCKED",
                code="large-tracked-file",
                path=relative_path,
                message=f"tracked file is {size} bytes; max public-release size is {max_bytes} bytes",
            )
        )

    return findings


def check_dinov3_source_policy(repo_root: Path) -> list[PublicReleaseFinding]:
    dinov3_path = repo_root / "fusion_zero_shot/src/dino/third_party/dinov3"
    if not dinov3_path.exists():
        return []

    findings: list[PublicReleaseFinding] = []
    license_path = dinov3_path / "LICENSE.md"
    if not license_path.exists():
        findings.append(
            PublicReleaseFinding(
                level="BLOCKED",
                code="dinov3-license-missing",
                path="fusion_zero_shot/src/dino/third_party/dinov3/LICENSE.md",
                message="bundled DINOv3 source must keep upstream LICENSE.md",
            )
        )

    strategy = repo_root / "docs/dinov3_source_strategy.md"
    if not strategy.exists():
        findings.append(
            PublicReleaseFinding(
                level="WARN",
                code="dinov3-strategy-doc-missing",
                path="docs/dinov3_source_strategy.md",
                message="DINOv3 bundled/external source policy is not documented",
            )
        )

    return findings


def check_third_party_notice(repo_root: Path) -> list[PublicReleaseFinding]:
    return check_dinov3_source_policy(repo_root)


def check_public_release(repo_root: Path) -> list[PublicReleaseFinding]:
    findings: list[PublicReleaseFinding] = []
    findings.extend(check_private_paths(repo_root))
    findings.extend(check_gitignore_entry(repo_root, "docs/superpowers/"))
    findings.extend(check_tracked_file_sizes(repo_root))
    findings.extend(check_dinov3_source_policy(repo_root))
    return findings
