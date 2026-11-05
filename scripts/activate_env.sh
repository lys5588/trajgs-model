# Source this file from bash: source scripts/activate_env.sh [trajgs|unimotion]
_trajgs_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_trajgs_name="${1:-trajgs}"
if [[ "$_trajgs_name" != trajgs && "$_trajgs_name" != unimotion ]]; then
    echo 'Usage: source scripts/activate_env.sh [trajgs|unimotion]' >&2
    return 1
fi
if [[ -f "$_trajgs_repo/.local/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$_trajgs_repo/.local/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo 'Conda is unavailable. Install Miniforge first; see docs/bootstrap.md.' >&2
    return 1
fi
conda activate "$_trajgs_name" || return 1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCH_EXTENSIONS_DIR="$_trajgs_repo/.cache/torch_extensions/$_trajgs_name"
export TRITON_CACHE_DIR="$_trajgs_repo/.cache/triton/$_trajgs_name"
unset _trajgs_repo _trajgs_name
