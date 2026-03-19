#!/bin/bash

#SBATCH --job-name=bagging_optuna
#SBATCH --output=logs/bagging_%j.out
#SBATCH --error=logs/bagging_%j.err
#SBATCH --time=23:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# 1. Limpieza de módulos heredados
module purge

# 2. Módulos necesarios (CPU only)
module load Miniforge3

# 3. Python absoluto del entorno
PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

# 4. Proyecto (para que las rutas relativas coincidan con los scripts)
PROJECT_DIR=/scratch/aitordiez/SMARTCO-DeliveryStatus-Prediction
DB_PATH=$PROJECT_DIR/output/optuna/bagging_study.db

# 5. Crear directorios necesarios
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$(dirname "$DB_PATH")"

cd "$PROJECT_DIR" || exit 1

# 6. Hilos CPU (ayuda a BLAS/numexpr, etc.)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# 8. Diagnóstico
echo "=== Diagnóstico de Entorno ==="
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodo:              $(hostname)"
echo "Start time:        $(date)"
echo "Working dir:       $(pwd)"
echo "CPUs per task:     ${SLURM_CPUS_PER_TASK:-1}"
echo "Python:            $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version:    $($PYTHON_BIN --version)"
echo "Optuna DB (expected): $DB_PATH"
echo ""

# 9. Lanzar tuning
echo "=== Iniciando Optuna Bagging tuning ==="
$PYTHON_BIN -u src/09_Bagging.py $EXTRA_ARGS

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

