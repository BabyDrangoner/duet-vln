#!/usr/bin/env bash
# Run from a foreground terminal or tmux. The pipeline handles SIGINT/SIGTERM.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/third_party/Matterport3DSimulator/build${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u -m vln_improve.pipeline "$@"
