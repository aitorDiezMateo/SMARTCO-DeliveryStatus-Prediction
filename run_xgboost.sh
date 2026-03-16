#!/bin/bash

#SBATCH --job-name=xgboost_optuna
#SBATCH --output=logs/xgboost_%j.out
#SBATCH --error=logs/xgboost_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# ---------------------------------------------------------------------------
# Optuna DB path — stored in the project directory so persists between jobs.
# You can override from the command line:
#   sbatch run_xgboost.sh --db /path/to/study.db --n-trials 200
# ---------------------------------------------------------------------------
DB_PATH=/scratch/aitordiez/SMARTCO-DeliveryStatus-Prediction/output/optuna/xgboost_study.db
N_TRIALS=100

# Pass any extra CLI args forwarded via sbatch (e.g. --n-trials 50)
EXTRA_ARGS="$@"

# 1. Limpieza de módulos heredados
module purge

# 2. Módulos necesarios
module load Miniforge3
module load CUDA/12.8.0

# 3. Python absoluto del entorno
PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

# 4. Crear directorios necesarios
mkdir -p logs
mkdir -p "$(dirname "$DB_PATH")"

# 5. Diagnóstico
echo "=== Diagnóstico de Entorno ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodo:              $(hostname)"
echo "Start time:        $(date)"
echo "GPU assigned:      $CUDA_VISIBLE_DEVICES"
echo "Working dir:       $(pwd)"
echo "Python:            $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PYTHON_BIN --version)"
echo "XGBoost version:   $($PYTHON_BIN -c 'import xgboost as xgb; print(xgb.__version__)')"
echo "USE_CUDA:          $($PYTHON_BIN -c 'import xgboost as xgb; print(xgb.build_info().get("USE_CUDA"))')"
echo "Optuna DB:         $DB_PATH"
echo "N trials:          $N_TRIALS"
echo ""

# 6. Estado GPU
echo "=== nvidia-smi ==="
nvidia-smi
echo ""

# 7. Lanzar tuning
echo "=== Iniciando Optuna XGBoost tuning ==="
$PYTHON_BIN -u src/07_XGBoost.py \
    --db "$DB_PATH" \
    --n-trials "$N_TRIALS" \
    $EXTRA_ARGS

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "RESULTADO: Tuning completado correctamente"
    echo "DB:        $DB_PATH"
    echo "End time:  $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "RESULTADO: Fallo (código $EXIT_CODE)"
    echo "End time:  $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE
