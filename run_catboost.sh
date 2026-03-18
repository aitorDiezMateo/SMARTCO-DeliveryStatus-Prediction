#!/bin/bash

#SBATCH --job-name=catboost_optuna
#SBATCH --output=logs/catboost_%j.out
#SBATCH --error=logs/catboost_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# ---------------------------------------------------------------------------
# Optuna tuning for CatBoost (GPU by default).
# Stored DB persists between jobs in the project directory.
# You can still pass extra args via sbatch and they will be forwarded:
#   sbatch run_catboost.sh --some-flag
# ---------------------------------------------------------------------------

# Project layout (keep absolute paths consistent with other run_*.sh scripts)
PROJECT_DIR=/scratch/aitordiez/SMARTCO-DeliveryStatus-Prediction
DB_PATH=$PROJECT_DIR/output/optuna/catboost_study.db

# Pass any extra CLI args forwarded via sbatch
EXTRA_ARGS="$@"

# 1. Limpieza de módulos heredados
module purge

# 2. Módulos necesarios (CPU only)
module load Miniforge3
module load CUDA/12.8.0

# 3. Python absoluto del entorno
PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

# 4. Crear directorios necesarios
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$DB_PATH")"

cd "$PROJECT_DIR" || exit 1

# 5. Hilos CPU (ayuda a BLAS/numexpr, etc.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 6. Diagnóstico
echo "=== Diagnóstico de Entorno ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodo:              $(hostname)"
echo "Start time:        $(date)"
echo "GPU assigned:      $CUDA_VISIBLE_DEVICES"
echo "Working dir:       $(pwd)"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Python:            $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PYTHON_BIN --version)"
echo "sklearn version:   $($PYTHON_BIN -c 'import sklearn; print(sklearn.__version__)')"
echo "optuna version:    $($PYTHON_BIN -c 'import optuna; print(optuna.__version__)')"
echo "catboost version:  $($PYTHON_BIN -c 'import catboost as cb; print(cb.__version__)')"
echo "Optuna DB (expected): $DB_PATH"
echo ""

# 7. Estado GPU
echo "=== nvidia-smi ==="
nvidia-smi
echo ""

# 7. Lanzar tuning
echo "=== Iniciando Optuna CatBoost tuning ==="
export CATBOOST_TASK_TYPE="${CATBOOST_TASK_TYPE:-GPU}"
export CATBOOST_DEVICES="${CATBOOST_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
$PYTHON_BIN -u src/08_CatBoost.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "RESULTADO: Tuning completado correctamente"
    echo "DB (expected): $DB_PATH"
    echo "End time:      $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "RESULTADO: Fallo (código $EXIT_CODE)"
    echo "End time:  $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

