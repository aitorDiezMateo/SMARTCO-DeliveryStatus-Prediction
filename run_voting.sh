#!/bin/bash

#SBATCH --job-name=voting
#SBATCH --output=logs/voting_%j.out
#SBATCH --error=logs/voting_%j.err
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# ---------------------------------------------------------------------------
# Run voting using:
# - src/11_Voting.py
#
# Voting mode can be set via env var:
#   export VOTING_MODE=soft   # default
#   export VOTING_MODE=hard
#
# Assumes Optuna study DBs already exist under `output/optuna/`.
# ---------------------------------------------------------------------------

EXTRA_ARGS="$@"

PROJECT_DIR=/scratch/aitordiez/SMARTCO-DeliveryStatus-Prediction

module purge
module load Miniforge3
module load CUDA/12.8.0

PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== Diagnóstico de Entorno ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodo:              $(hostname)"
echo "Start time:        $(date)"
echo "GPU assigned:      $CUDA_VISIBLE_DEVICES"
echo "Working dir:       $(pwd)"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Python:            $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PYTHON_BIN --version)"
echo "VOTING_MODE:       ${VOTING_MODE:-soft}"
echo ""

echo "=== nvidia-smi ==="
nvidia-smi
echo ""

export CATBOOST_TASK_TYPE="${CATBOOST_TASK_TYPE:-GPU}"
export CATBOOST_DEVICES="${CATBOOST_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "=== Iniciando src/11_Voting.py ==="
$PYTHON_BIN -u src/11_Voting.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "RESULTADO: Voting completado correctamente"
    echo "End time:  $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "RESULTADO: Fallo (código $EXIT_CODE)"
    echo "End time:  $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

