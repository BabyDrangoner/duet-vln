#!/usr/bin/env bash
set -uo pipefail
cd /home/xxl/projects/duet-vln
source outputs/runtime-wsl/activate.sh
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_stage() {
  local stage_name="$1"
  shift
  "$@"
  local stage_exit=$?
  printf '{"stage":"%s","exit_code":%d}\n' "$stage_name" "$stage_exit" >> outputs/runtime-wsl/acceptance-stages.jsonl
  if (( stage_exit != 0 )); then exit "$stage_exit"; fi
}
run_stage asset_verification .venv/bin/python scripts/download_assets.py --verify-only
run_stage navigation_identity .venv/bin/python scripts/run_duet.py --mode identity --split train_fit --limit 2 --output outputs/runtime-wsl/wsl-identity-smoke.json
run_stage collect_resume_cache .venv/bin/python scripts/run_duet.py --mode collect --split train_fit --limit 8 --cache outputs/wsl-migration-cache-fit8 --output outputs/runtime-wsl/wsl-cache-fit8-report.json
run_stage cuda_resume .venv/bin/python scripts/smoke_resume.py --run-id wsl-migration-resume-20261006 --backup-backend filesystem --backup-root /home/xxl/vln-backups --cache outputs/wsl-migration-cache-fit8
