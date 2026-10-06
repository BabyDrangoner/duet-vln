#!/usr/bin/env bash
# Run inside WSL/Linux, preferably in tmux. SIGINT/SIGTERM save resumable state.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f outputs/runtime-wsl/activate.sh ]]; then
  source outputs/runtime-wsl/activate.sh
fi
if [[ "$(uname -s)" != Linux ]]; then
  echo "train_wsl.sh requires WSL/Linux." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; run scripts/setup_wsl.sh first." >&2
  exit 1
fi
export PYTHONPATH="$PWD/src:$PWD/third_party/Matterport3DSimulator/build${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ $# -eq 0 ]]; then
  set -- --config configs/pipeline_wsl.json
fi
exec .venv/bin/python -u -m vln_improve.pipeline "$@"
