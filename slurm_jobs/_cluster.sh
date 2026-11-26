# shellcheck shell=bash
# Load cluster.env from the repo root. Sourced by slurm_jobs/*.sh
_CLUSTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "${_CLUSTER_REPO_ROOT}/cluster.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${_CLUSTER_REPO_ROOT}/cluster.env"
  set +a
fi

EVENTS_DIR="${EVENTS_DIR:-${_CLUSTER_REPO_ROOT}}"
MODEL_DIR="${MODEL_DIR:-${EVENTS_DIR}/models}"
export EVENTS_DIR MODEL_DIR SLURM_ACCOUNT SLURM_MAIL_USER SLURM_GPU_CLUSTER
