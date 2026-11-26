#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/_cluster.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}/slurm_jobs"
fi
source "${SCRIPT_DIR}/_cluster.sh"
cd "${EVENTS_DIR}"
exec python3 "${SCRIPT_DIR}/run.py" "$@"
