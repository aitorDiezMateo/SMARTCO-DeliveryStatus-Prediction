#!/bin/bash

#SBATCH --job-name=best_models_individual
#SBATCH --output=logs/best_models_individual_%j.out
#SBATCH --error=logs/best_models_individual_%j.err
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# ---------------------------------------------------------------------------
# Run the best individual models:
# - src/12_best_models_individual.py
#
# This script loads the best Optuna params from:
# - output/optuna/xgboost_study.db
# - output/optuna/catboost_study.db
# - output/optuna/bagging_study.db
#
# and trains the models on full train, evaluates on test,
# saving predictions to output/predictions/.
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
echo ""

echo "=== nvidia-smi ==="
nvidia-smi
echo ""

export CATBOOST_TASK_TYPE="${CATBOOST_TASK_TYPE:-GPU}"
export CATBOOST_DEVICES="${CATBOOST_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "=== Iniciando src/12_best_models_individual.py ==="
$PYTHON_BIN -u src/12_best_models_individual.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "RESULTADO: Script completado correctamente"
    echo "End time:  $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "RESULTADO: Fallo (código $EXIT_CODE)"
    echo "End time:  $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

