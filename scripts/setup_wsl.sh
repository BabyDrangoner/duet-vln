#!/usr/bin/env bash
# Prepare this checkout on WSL2/Ubuntu. No driver installation and no training.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python_bin=""
install_system=0
download_assets=1
run_smoke=0
bootstrap_python=0
user_native_deps=0
jobs=2
while (($#)); do
  case "$1" in
    --python) python_bin="${2:?--python requires an executable}"; shift 2 ;;
    --install-system) install_system=1; shift ;;
    --skip-assets) download_assets=0; shift ;;
    --smoke) run_smoke=1; shift ;;
    --bootstrap-python) bootstrap_python=1; shift ;;
    --user-native-deps) user_native_deps=1; shift ;;
    --jobs) jobs="${2:?--jobs requires a positive integer}"; shift 2 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/setup_wsl.sh [--python python3.11] [--install-system]
                               [--bootstrap-python] [--user-native-deps]
                               [--skip-assets] [--smoke] [--jobs 2]

Prepare .venv, pinned DUET/MatterSim source, and SHA-verified public assets.
--install-system explicitly allows apt installation of missing build dependencies.
--bootstrap-python installs user-local uv and Python 3.11 when Python is unspecified.
--user-native-deps extracts missing GLM/OSMesa apt packages into this checkout, without sudo.
--skip-assets skips the 5.3 GB annotation/features/checkpoint download and graphs.
--smoke runs CUDA + MatterSim checks and a two-instruction train_fit identity rollout.
No full training or val_unseen evaluation is started by this script.
HELP
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo '--jobs must be a positive integer' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'Run this script inside WSL2 or Linux.' >&2; exit 2; }
cd "$repo_root"
mkdir -p outputs/runtime-wsl

packages=(build-essential cmake pkg-config git libjsoncpp-dev libglm-dev libosmesa6-dev libopencv-dev)
missing=()
for package in "${packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    missing+=("$package")
  fi
done
if ((${#missing[@]})); then
  if (( user_native_deps )); then
    for package in "${missing[@]}"; do
      case "$package" in
        libglm-dev|libosmesa6-dev) ;;
        *) echo "--user-native-deps supports GLM/OSMesa only; install missing $package through your administrator." >&2; exit 2 ;;
      esac
    done
    native_prefix="$repo_root/outputs/runtime-wsl/native"
    mkdir -p "$native_prefix/packages"
    apt-get --simulate install --no-install-recommends "${missing[@]}" > outputs/runtime-wsl/native-apt-plan.txt
    mapfile -t native_packages < <(awk '/^Inst / {print $2}' outputs/runtime-wsl/native-apt-plan.txt)
    ((${#native_packages[@]})) || { echo 'APT did not produce a dependency plan.' >&2; exit 2; }
    for package in "${native_packages[@]}"; do
      (cd "$native_prefix/packages" && apt-get download "$package")
    done
    for deb in "$native_prefix"/packages/*.deb; do dpkg-deb -x "$deb" "$native_prefix"; done
    export CPATH="$native_prefix/usr/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$native_prefix/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$native_prefix/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PKG_CONFIG_PATH="$native_prefix/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    # This package is relocated; system JSONCPP/OpenCV .pc files retain /usr.
    python3 - "$native_prefix" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for path in (root / 'usr/lib/x86_64-linux-gnu/pkgconfig').glob('*.pc'):
    text = path.read_text()
    text = text.replace('prefix=/usr\n', f'prefix={root}/usr\n')
    text = text.replace('includedir=/usr/include\n', f'includedir={root}/usr/include\n')
    text = text.replace('libdir=/usr/lib/x86_64-linux-gnu\n', f'libdir={root}/usr/lib/x86_64-linux-gnu\n')
    path.write_text(text)
PY
  elif (( ! install_system )); then
    printf 'Missing system packages: %s\n' "${missing[*]}" >&2
    echo 'Use --user-native-deps for GLM/OSMesa, --install-system for apt installation, or ask your administrator.' >&2
    exit 2
  else
    command -v apt-get >/dev/null || { echo '--install-system requires an apt-based distribution.' >&2; exit 2; }
    elevate=()
    if (( EUID != 0 )); then elevate=(sudo); fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y "${missing[@]}"
  fi
fi

if [[ -z "$python_bin" ]] && (( bootstrap_python )); then
  bootstrap_dir="$repo_root/outputs/runtime-wsl/bootstrap"
  if [[ ! -x "$bootstrap_dir/bin/python" ]]; then python3 -m venv "$bootstrap_dir"; fi
  "$bootstrap_dir/bin/python" -m pip install 'uv==0.12.5'
  export UV_PYTHON_INSTALL_DIR="$repo_root/outputs/runtime-wsl/python"
  "$bootstrap_dir/bin/uv" python install 3.11.16
  python_bin="$("$bootstrap_dir/bin/uv" python find 3.11.16)"
fi
if [[ -z "$python_bin" ]]; then
  if command -v python3.11 >/dev/null; then python_bin=python3.11
  elif [[ -x .venv/bin/python ]]; then python_bin="$repo_root/.venv/bin/python"
  else
    echo 'Use --bootstrap-python for a user-local Python 3.11, or pass --python /path/to/python3.11.' >&2
    exit 2
  fi
fi
"$python_bin" - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit('Use Python 3.11 (recommended) or 3.10; other versions are not validated for this pipeline.')
PY
if [[ ! -x .venv/bin/python ]]; then "$python_bin" -m venv .venv; fi
if [[ "$("$python_bin" -c 'import sys; print(sys.version_info[:2])')" != "$(.venv/bin/python -c 'import sys; print(sys.version_info[:2])')" ]]; then
  echo 'Existing .venv uses a different Python version. Preserve it and choose a fresh checkout.' >&2
  exit 2
fi
venv_python="$repo_root/.venv/bin/python"
export PATH="$repo_root/.venv/bin:$PATH"
"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
"$venv_python" -m pip install -e '.[duet,test]' 'numpy==1.26.4'
"$venv_python" scripts/prepare_duet.py

native="$repo_root/third_party/Matterport3DSimulator"
native_commit=589d091b111333f9e9f9d6cfd021b2eb68435925
prepare_git_source() {
  local destination="$1" source_url="$2" pinned_commit="$3"
  if [[ ! -e "$destination/.git" ]]; then
    if [[ -d "$destination" && -n "$(ls -A "$destination")" ]]; then
      echo "Existing source is not a Git checkout: $destination; preserve it and use a fresh checkout." >&2
      exit 2
    fi
    git init "$destination"
    git -C "$destination" remote add origin "$source_url"
  fi
  [[ "$(git -C "$destination" remote get-url origin)" == "$source_url" ]] || { echo "Unexpected source remote: $destination" >&2; exit 2; }
  if ! git -C "$destination" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$destination" fetch --depth 1 origin "$pinned_commit"
    git -C "$destination" checkout --detach FETCH_HEAD
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$pinned_commit" ]] || { echo "Unexpected source version at $destination; preserve it and use a fresh checkout." >&2; exit 2; }
}
prepare_git_source "$native" https://github.com/peteanderson80/Matterport3DSimulator.git "$native_commit"
"$venv_python" - "$native" <<'PY'
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
patches = {
    'CMakeLists.txt': ('cmake_minimum_required(VERSION 2.8)', 'cmake_minimum_required(VERSION 3.10)'),
    'src/lib/NavGraph.cpp': ('CV_LOAD_IMAGE_ANYDEPTH', 'cv::IMREAD_ANYDEPTH'),
}
for name, (old, new) in patches.items():
    original = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{name}'])
    if old.encode() not in original:
        raise SystemExit(f'MatterSim compatibility patch anchor missing: {name}')
    prepared = original.replace(old.encode(), new.encode())
    target = root / name
    if target.read_bytes() not in (original, prepared):
        raise SystemExit(f'Refusing to overwrite local MatterSim edits: {target}')
    target.write_bytes(prepared)
modified = set(subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', 'HEAD']).decode().splitlines())
if modified - set(patches) - {'pybind11'}:
    raise SystemExit(f'Unexpected local MatterSim source modifications: {sorted(modified)}')
PY
# Upstream v2.13.6, pinned by commit as well as its release name.
prepare_git_source "$native/pybind11" https://github.com/pybind/pybind11.git a2e59f0e7065404b44dfe92a28aca47ba1378dc4
[[ -z "$(git -C "$native/pybind11" status --porcelain --untracked-files=no)" ]] || { echo 'Refusing modified pybind11 sources.' >&2; exit 2; }
cmake -S "$native" -B "$native/build" -DOSMESA_RENDERING=ON -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$venv_python"
cmake --build "$native/build" --target MatterSimPython -j "$jobs"
export PYTHONPATH="$native/build:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
# Reusable activation includes relocated OSMesa runtime libraries when present.
"$venv_python" - "$repo_root" <<'PY'
from pathlib import Path
import shlex
import sys
root = Path(sys.argv[1])
prefix = root / 'outputs/runtime-wsl/native'
lines = ['# Generated by scripts/setup_wsl.sh; source this file from bash.']
values = {'PATH': str(root / '.venv/bin'),
          'PYTHONPATH': f'{root}/third_party/Matterport3DSimulator/build:{root}/src'}
if prefix.is_dir():
    values.update({'CPATH': f'{prefix}/usr/include', 'LIBRARY_PATH': f'{prefix}/usr/lib/x86_64-linux-gnu',
                   'LD_LIBRARY_PATH': f'{prefix}/usr/lib/x86_64-linux-gnu',
                   'PKG_CONFIG_PATH': f'{prefix}/usr/lib/x86_64-linux-gnu/pkgconfig'})
for key, value in values.items():
    lines.append(f'export {key}={shlex.quote(value)}${{{key}:+:${key}}}')
lines.append('export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8')
(root / 'outputs/runtime-wsl/activate.sh').write_text('\n'.join(lines) + '\n')
PY
"$venv_python" - <<'PY'
import MatterSim
import torch
if str(torch.__version__) != '2.5.1+cu121':
    raise SystemExit(f'Unexpected PyTorch build: {torch.__version__}')
print('MatterSim import passed; PyTorch', torch.__version__)
PY
if (( download_assets )); then "$venv_python" scripts/download_assets.py; fi
"$venv_python" -m pip freeze > outputs/runtime-wsl/requirements-freeze.txt
git rev-parse HEAD > outputs/runtime-wsl/source-commit.txt 2>/dev/null || true
if (( run_smoke )); then
  "$venv_python" scripts/preflight.py > outputs/runtime-wsl/preflight.json
  "$venv_python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in WSL; check the Windows NVIDIA driver and WSL2 GPU access.'
x = torch.arange(16, dtype=torch.float32, device='cuda')
assert x.square().sum().item() == 1240
print('CUDA arithmetic passed:', torch.cuda.get_device_name(0))
PY
  smoke_output="outputs/runtime-wsl/identity-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
  "$venv_python" scripts/run_duet.py --mode identity --split train_fit --limit 2 --output "$smoke_output"
fi
echo 'WSL environment prepared. See docs/wsl.md for activation, resume protection, and smoke commands.'
