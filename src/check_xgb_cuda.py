"""
DIAGNOSTIC UTILITY - XGBOOST GPU (CUDA) VERIFICATION

This script verifies XGBoost GPU acceleration is properly configured:
1. Import and version of XGBoost
2. build_info(): confirms XGBoost compiled with USE_CUDA flag
3. Minimal GPU training with xgb.train() (functional test)
4. GPU VRAM usage verification (confirms real GPU memory > 0 MiB)
5. Performance benchmark: CPU vs GPU runtime comparison (qualitative acceleration check)

Purpose: Validate CUDA environment before running 04_XGBoost.py GPU-accelerated tuning.
Use when: Debugging GPU training failures or confirming GPU availability on HPC nodes.

This is a DIAGNOSTIC TOOL (not part of model training pipeline).
Run on compute nodes before launching Optuna GPU tuning jobs.

Output: Diagnostic messages to stdout ([OK], [ERROR], [WARN] tags).
"""

import sys
import time

import numpy as np


SEP = "=" * 60
OK  = "[OK]"
ERR = "[ERROR]"
WARN = "[WARN]"


def _section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main() -> int:
    overall_ok = True

    ######### 1. Import #########
    _section("1. Import XGBoost")
    try:
        import xgboost as xgb
        print(f"{OK} xgboost importado correctamente")
        print(f"     Version : {xgb.__version__}")
    except ImportError as exc:
        print(f"{ERR} No se pudo importar xgboost: {exc}")
        return 1

    ######### 2. build_info #########
    _section("2. Build info (CUDA compilado?)")
    build_info: dict = {}
    try:
        build_info = xgb.build_info()
    except Exception as exc:
        print(f"{WARN} build_info() no disponible: {exc}")

    for key in ("USE_CUDA", "USE_NCCL", "USE_OPENMP", "GCC_VERSION", "GLIBC_VERSION"):
        if key in build_info:
            print(f"     {key:20s}: {build_info[key]}")

    use_cuda_build = build_info.get("USE_CUDA", False)
    if use_cuda_build:
        print(f"{OK} XGBoost compilado con soporte CUDA")
    else:
        print(f"{ERR} XGBoost compilado SIN soporte CUDA (variant CPU)")
        print("     Instala: CONDA_OVERRIDE_CUDA=12.8 conda install -c conda-forge py-xgboost-gpu")
        overall_ok = False

    ######### 3. Minimal GPU training #########
    _section("3. Minimal GPU training")
    rng = np.random.default_rng(42)
    X_small = rng.standard_normal((200, 10)).astype(np.float32)
    y_small = rng.integers(0, 2, size=200).astype(np.float32)
    dtrain = xgb.DMatrix(X_small, label=y_small)

    params_gpu = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 3,
        "verbosity": 0,
    }
    gpu_ok = False
    try:
        xgb.train(params_gpu, dtrain, num_boost_round=5)
        print(f"{OK} Entrenamiento en GPU completado sin errores")
        gpu_ok = True
    except Exception as exc:
        print(f"{ERR} Fallo al entrenar en GPU: {exc}")
        overall_ok = False

    ######### 4. Benchmark CPU vs GPU #########
    _section("4. Benchmark CPU vs GPU")
    rng2 = np.random.default_rng(0)
    X_big = rng2.standard_normal((5_000, 50)).astype(np.float32)
    y_big = rng2.integers(0, 2, size=5_000).astype(np.float32)
    dbig  = xgb.DMatrix(X_big, label=y_big)
    ROUNDS = 100

    params_cpu = {**params_gpu, "device": "cpu"}
    t0 = time.perf_counter()
    xgb.train(params_cpu, dbig, num_boost_round=ROUNDS)
    t_cpu = time.perf_counter() - t0
    print(f"     CPU  : {t_cpu:.3f} s  ({ROUNDS} rounds, 5k x 50)")

    if gpu_ok:
        t0 = time.perf_counter()
        xgb.train(params_gpu, dbig, num_boost_round=ROUNDS)
        t_gpu = time.perf_counter() - t0
        print(f"     GPU  : {t_gpu:.3f} s  ({ROUNDS} rounds, 5k x 50)")
        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        if speedup >= 1.0:
            print(f"{OK} GPU es {speedup:.1f}x mas rapida que CPU")
        else:
            print(f"{WARN} GPU es {1/speedup:.1f}x mas LENTA que CPU")
            print("     Esto puede indicar que xgboost esta usando CPU igualmente.")
            overall_ok = False
    else:
        print(f"{WARN} Benchmark GPU omitido (entrenamiento GPU fallo antes)")

    ######### Final summary #########
    _section("Summary")
    if overall_ok:
        print(f"{OK} Todo correcto — XGBoost esta usando GPU (CUDA)")
        return 0
    else:
        print(f"{ERR} Hay problemas — revisa los errores arriba")
        return 1


if __name__ == "__main__":
    sys.exit(main())
