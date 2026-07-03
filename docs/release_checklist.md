# Public Release Checklist

Use this checklist before pushing a public paper-release branch.

First review `docs/release_status.md` for the current runnable surface,
accepted warnings, contract coverage, and open gaps.

## Private Paths

- Run `bash scripts/check_public_release.sh`.
- Treat `private-path` findings as blocking.
- Keep public placeholders such as `/path/to/Probing_Bridging_Affordance`.
- Do not commit machine paths such as `/Users/...`, `/home/...`, `/mnt/...`, or `/data/...`.

## Large Tracked Files

- Treat `large-tracked-file` findings as blocking.
- Keep datasets, checkpoints, generated caches, and model weights out of git.
- Store required assets under ignored local directories or document external download/setup steps.

## Ignored Local Agent Files

- Keep `docs/superpowers/` in `.gitignore`.
- Do not push local planning, agent memory, worktree, or generated scratch files.
- Keep `.worktrees/` ignored.

## Third-Party Source

- DINOv3 source currently exists under `fusion_zero_shot/src/dino/third_party/dinov3`.
- Keep its upstream license file with the bundled source tree.
- Before public release, decide whether to retain the bundled DINOv3 tree with notice or replace it with a user-provided `models/dinov3/` source convention.
- The registered gap is `dinov3-external-source`.
- Source policy is documented in `docs/dinov3_source_strategy.md`.

## Missing Development Code

These gaps are intentional until true development code is recovered:

- `interaction-batch-eval`
- `fusion-cache-build`
- `dinov3-external-source`
- `sd21-local-model`
- `umd-pytorch-dataset-migration`
- `geometry-heavy-runtime-migration`
- `fusion-heavy-runtime-migration`
- `interaction-heavy-runtime-migration`

Do not fabricate missing experiment logic to close these gaps.

## Required Commands

```bash
bash scripts/check_public_release.sh
python -m pba.audit --config configs/reproduce/main.yaml --skip-asset-check --show-gaps --public-release
python -m pba.reproduce main --config configs/reproduce/main.yaml --skip-asset-check --dry-run
python -m pytest -q
```
