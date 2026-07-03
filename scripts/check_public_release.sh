#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/reproduce/main.yaml"

python -m pba.audit \
  --config "$CONFIG" \
  --skip-asset-check \
  --show-gaps \
  --public-release

python -m pba.reproduce main \
  --config "$CONFIG" \
  --skip-asset-check \
  --dry-run

python -m pytest -q \
  tests/test_public_release_checks.py \
  tests/test_release_audit.py \
  tests/test_release_status_docs.py \
  tests/test_release_docs.py \
  tests/test_packaging_metadata.py \
  tests/test_fusion_cache_contract.py \
  tests/test_fusion_cache_docs.py \
  tests/test_interaction_batch_contract.py \
  tests/test_interaction_batch_docs.py \
  tests/test_sd21_model_contract.py \
  tests/test_sd21_model_docs.py \
  tests/test_umd_dataset_contract.py \
  tests/test_umd_dataset_docs.py \
  tests/test_geometry_runtime_contract.py \
  tests/test_geometry_runtime_docs.py \
  tests/test_fusion_runtime_contract.py \
  tests/test_fusion_runtime_docs.py \
  tests/test_interaction_runtime_contract.py \
  tests/test_interaction_runtime_docs.py \
  tests/test_public_api_smoke.py
