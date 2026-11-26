"""PKL vs ONNX runtime benchmark for the Random Forest deployment candidate.

Fair-comparison rules (v2):
  * both runtimes pinned to ONE thread (OMP/MKL/OpenBLAS env + ORT SessionOptions);
  * single-sample latency cycles through the 440 real outer-test rows;
  * batch = 1,000 rows drawn with replacement from the outer-test rows (seed 42);
  * prediction agreement checked on all 440 outer-test rows;
  * deployed crop_model.onnx is checked against model/research_best_model.pkl;
  * the Stacking ensemble (best held-out accuracy) is timed under scikit-learn for context.
Outputs: paper/assets/onnx_runtime_results.json
"""
import os

for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"

import json
import pickle
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import shap
import lime
import skl2onnx
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

SEED = 42
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DATASET_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "atharvaingle" / "crop-recommendation-dataset" / "versions" / "1" / "Crop_recommendation.csv"
BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
MODEL_DIR = BASE / "model"
RESULTS_JSON = ASSETS / "onnx_runtime_results.json"
RF_PKL = MODEL_DIR / "runtime_rf_pipeline.pkl"
RF_ONNX = MODEL_DIR / "runtime_rf_pipeline.onnx"
DEPLOYED_ONNX = BASE / "crop_model.onnx"
RESEARCH_PKL = MODEL_DIR / "research_best_model.pkl"

N_WARMUP, N_LOAD, N_SINGLE, N_BATCH, BATCH_SIZE = 50, 50, 2000, 100, 1000


def ort_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def rf_pipeline():
    return Pipeline([("minmax", MinMaxScaler()), ("standard", StandardScaler()),
                     ("clf", RandomForestClassifier(n_estimators=700, random_state=SEED, n_jobs=1))])


def stacking_pipeline():
    # tuned configuration from paper/assets/tuned_ensembles.json (Stacking, LR meta)
    bp = json.loads((ASSETS / "tuned_ensembles.json").read_text())["results"][0]["best_params"]
    return Pipeline([("minmax", MinMaxScaler()), ("standard", StandardScaler()), ("clf", StackingClassifier(
        estimators=[("rf", RandomForestClassifier(n_estimators=bp["rf_n"], random_state=SEED, n_jobs=1)),
                    ("et", ExtraTreesClassifier(n_estimators=bp["et_n"], random_state=SEED, n_jobs=1)),
                    ("svc", SVC(kernel="rbf", C=bp["svc_C"], probability=True, random_state=SEED)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=tuple(bp["mlp_h"]), max_iter=400, random_state=SEED))],
        final_estimator=LogisticRegression(C=bp["meta_C"], max_iter=2000, random_state=SEED),
        cv=3, n_jobs=1, stack_method="predict_proba"))])


def export(pipeline, path):
    m = convert_sklearn(pipeline, initial_types=[("float_input", FloatTensorType([None, len(FEATURES)]))],
                        options={id(pipeline): {"zipmap": False}})
    path.write_bytes(m.SerializeToString())


def timeit(fn, n):
    out = []
    for i in range(n):
        t0 = time.perf_counter(); fn(i); out.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(out)), float(np.std(out)), float(np.median(out))


def onnx_labels(sess, X):
    return np.asarray(sess.run(["label"], {"float_input": X.astype(np.float32)})[0]).reshape(-1)


def main():
    df = pd.read_csv(DATASET_CACHE)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    rows = [Xte[i:i + 1] for i in range(len(Xte))]
    batch = Xte[np.random.default_rng(SEED).integers(0, len(Xte), BATCH_SIZE)]

    rf = rf_pipeline().fit(Xtr, ytr)
    RF_PKL.write_bytes(pickle.dumps(rf))
    export(rf, RF_ONNX)

    pkl_load = timeit(lambda i: pickle.loads(RF_PKL.read_bytes()), N_LOAD)
    onnx_load = timeit(lambda i: ort_session(RF_ONNX), N_LOAD)

    sess = ort_session(RF_ONNX)
    for i in range(N_WARMUP):
        rf.predict(rows[i % len(rows)]); onnx_labels(sess, rows[i % len(rows)])
    pkl_single = timeit(lambda i: rf.predict(rows[i % len(rows)]), N_SINGLE)
    onnx_single = timeit(lambda i: sess.run(["label"], {"float_input": rows[i % len(rows)]}), N_SINGLE)
    pkl_batch = timeit(lambda i: rf.predict(batch), N_BATCH)
    onnx_batch = timeit(lambda i: sess.run(["label"], {"float_input": batch}), N_BATCH)

    pkl_test = rf.predict(Xte)
    onnx_test = onnx_labels(sess, Xte).astype(str)
    agree = int(np.sum(pkl_test.astype(str) == onnx_test))

    # Deployed artifact vs research winner (integer labels, sorted crop names)
    deployed = {}
    if DEPLOYED_ONNX.exists() and RESEARCH_PKL.exists():
        res = pickle.loads(RESEARCH_PKL.read_bytes())
        mm, sd = res["scalers"]["minmax"], res["scalers"]["standard"]
        res_pred = np.array(res["labels"])[res["model"].predict(sd.transform(mm.transform(Xte)))]
        dep_raw = onnx_labels(ort_session(DEPLOYED_ONNX), Xte)
        dep_pred = dep_raw.astype(str) if dep_raw.dtype.kind in "USO" else np.array(res["labels"])[dep_raw.astype(int)]
        deployed = {"deployed_vs_research_agreement": int(np.sum(dep_pred == res_pred)),
                    "deployed_test_accuracy": float(np.mean(dep_pred == yte)),
                    "deployed_equals_runtime_rf_file": DEPLOYED_ONNX.read_bytes() == RF_ONNX.read_bytes()}

    # Stacking (best held-out accuracy) latency under scikit-learn, same single-thread rules
    stk = stacking_pipeline().fit(Xtr, ytr)
    for i in range(N_WARMUP):
        stk.predict(rows[i % len(rows)])
    stk_single = timeit(lambda i: stk.predict(rows[i % len(rows)]), 500)
    stk_batch = timeit(lambda i: stk.predict(batch), 20)
    stk_acc = float(np.mean(stk.predict(Xte) == yte))

    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    payload = {
        "benchmark_model": "Random Forest (700 trees) + MinMax + Standard scaling, single pipeline",
        "environment": {"cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
                        "threads": 1, "scikit_learn": sklearn.__version__, "onnxruntime": ort.__version__,
                        "skl2onnx": skl2onnx.__version__, "onnx": onnx.__version__, "numpy": np.__version__,
                        "shap": shap.__version__, "lime": getattr(lime, "__version__", "0.2.0.1")},
        "repetitions": {"warmup": N_WARMUP, "load": N_LOAD, "single": N_SINGLE, "batch": N_BATCH, "batch_size": BATCH_SIZE},
        "load_ms": {"pkl": pkl_load, "onnx": onnx_load},
        "single_ms": {"pkl": pkl_single, "onnx": onnx_single},
        "batch_1000_ms": {"pkl": pkl_batch, "onnx": onnx_batch},
        "note_ms_tuple": "(mean, std, median)",
        "speedup": {"load": pkl_load[0] / onnx_load[0], "single": pkl_single[0] / onnx_single[0],
                    "batch_1000": pkl_batch[0] / onnx_batch[0]},
        "agreement_test_rows": {"agree": agree, "n": int(len(Xte))},
        "rf_test_accuracy": float(np.mean(pkl_test == yte)),
        "file_size_bytes": {"pkl": RF_PKL.stat().st_size, "onnx": RF_ONNX.stat().st_size},
        "deployed_check": deployed,
        "stacking_sklearn": {"single_ms": stk_single, "batch_1000_ms": stk_batch, "test_accuracy": stk_acc},
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
