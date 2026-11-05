#!/usr/bin/env bash
# Run from any directory after installing Miniforge/Conda.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$repo/.local/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$repo/.local/miniforge3/etc/profile.d/conda.sh"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi
cd "$repo"
if [[ $# -eq 0 ]]; then
    set -- unimotion trajgs
fi
for env_name in "$@"; do
    case "$env_name" in
        unimotion) python_version=3.10; torch_version=2.5.1; vision_version=0.20.1; cuda_index=cu124 ;;
        trajgs) python_version=3.12; torch_version=2.15.0; vision_version=0.29.1; cuda_index=cu132 ;;
        *) echo "Unknown environment: $env_name" >&2; exit 1 ;;
    esac
    if ! conda run -n "$env_name" python --version >/dev/null 2>&1; then
        conda create -y -n "$env_name" "python=$python_version" pip 'setuptools<81' wheel
    fi
    conda activate "$env_name"
    python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$python_version'.split('.'))), sys.version"
    python -m pip install 'setuptools<81' wheel
    python -m pip install "torch==$torch_version" "torchvision==$vision_version" \
        --index-url "https://download.pytorch.org/whl/$cuda_index"
    DS_BUILD_OPS=0 python -m pip install --no-build-isolation \
        -r "environments/requirements-$env_name.txt"
    python -m spacy download en_core_web_sm
    python scripts/check_dependencies.py
    WANDB_MODE=offline python scripts/check_environment.py
    conda deactivate
done
