#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/reproduce/main.yaml"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

python -m pba.audit --config "$CONFIG" "${ARGS[@]}"
