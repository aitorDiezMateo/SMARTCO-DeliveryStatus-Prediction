#!/bin/bash

#SBATCH --job-name=stacking
#SBATCH --output=logs/stacking_%j.out
#SBATCH --error=logs/stacking_%j.err
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# ---------------------------------------------------------------------------
# Run stacking baseline using:
# - src/10_stacking.py
# Assumes the Optuna study DBs already exist under `output/optuna/`.
# You can override with:
#   sbatch run_stacking.sh --some-arg
# ---------------------------------------------------------------------------

EXTRA_ARGS="$@"

PROJECT_DIR=/scratch/aitordiez/SMARTCO-DeliveryStatus-Prediction

# 1. Limpieza de módulos heredados
module purge

# 2. Módulos necesarios
module load Miniforge3
module load CUDA/12.8.0

# 3. Python absoluto del entorno
PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

# 4. Crear directorios necesarios
mkdir -p "$PROJECT_DIR/logs"

cd "$PROJECT_DIR" || exit 1

# 5. Hilos CPU (ayuda a BLAS/numexpr, etc.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 6. Estado GPU / diagnóstico
echo "=== Diagnóstico de Entorno ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodo:              $(hostname)"
echo "Start time:        $(date)"
echo "GPU assigned:      $CUDA_VISIBLE_DEVICES"
echo "Working dir:       $(pwd)"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Python:            $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PYTHON_BIN --version)"
echo "stacking script:  src/10_stacking.py"
echo ""

echo "=== nvidia-smi ==="
nvidia-smi
echo ""

# 7. Variables CatBoost (por si quieres forzar CPU/GPU)
export CATBOOST_TASK_TYPE="${CATBOOST_TASK_TYPE:-GPU}"
export CATBOOST_DEVICES="${CATBOOST_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

# 8. Lanzar stacking
echo "=== Iniciando src/10_stacking.py ==="
$PYTHON_BIN -u src/10_stacking.py $EXTRA_ARGS

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "RESULTADO: Stacking completado correctamente"
    echo "End time:  $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "RESULTADO: Fallo (código $EXIT_CODE)"
    echo "End time:  $(date)"
    echo "=========================================="
fi

exit $EXIT_CODE

