#!/bin/bash

#SBATCH --job-name=check_xgb_cuda
#SBATCH --output=logs/check_xgb_cuda_%j.out
#SBATCH --error=logs/check_xgb_cuda_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aitor.diez@opendeusto.es

# 1. LIMPIEZA DE ENTORNO
# Eliminamos cualquier módulo heredado (especialmente PyTorch/Python del sistema)
module purge

# 2. CARGA DE MÓDULOS ESPECÍFICOS (según tu 'module avail')
module load Miniforge3
module load CUDA/12.8.0

# 3. RUTA ABSOLUTA AL PYTHON DE TU ENTORNO
# Esto ignora cualquier otro Python que el sistema intente imponernos
PYTHON_BIN=/scratch/aitordiez/conda-env/smartco-delivery-status/bin/python

# 4. DIAGNÓSTICO (Para confirmar en el .out que todo es correcto)
echo "=== Diagnóstico de Entorno ==="
echo "Job ID:             $SLURM_JOB_ID"
echo "Nodo:               $(hostname)"
echo "Python esperado:    $PYTHON_BIN"
echo "Python detectado:   $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Versión Python:     $($PYTHON_BIN --version)"
echo ""

# 5. TEST DE GPU
echo "=== Estado de la GPU ==="
nvidia-smi
echo ""

# 6. EJECUCIÓN DEL SCRIPT
echo "=== Ejecutando Test XGBoost + CUDA ==="
# Usamos la ruta absoluta del binario para asegurar que use TU entorno
$PYTHON_BIN -u src/check_xgb_cuda.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "RESULTADO: Éxito. Entorno y GPU configurados correctamente."
else
    echo "RESULTADO: Fallo (Código $EXIT_CODE). Revisa los errores arriba."
fi

exit $EXIT_CODE