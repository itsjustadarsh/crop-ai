import json
import pickle
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "model"
ASSETS = BASE / "paper" / "assets"
BEST_MODEL_PKL = MODEL_DIR / "research_best_model.pkl"
ONNX_PATH = BASE / "crop_model.onnx"
OUT_JSON = ASSETS / "onnx_runtime_results.json"

N_WARMUP = 20
N_BENCH = 1000
N_LOAD = 100


def main():
    payload = pickle.load(open(BEST_MODEL_PKL, "rb"))
    pipeline = Pipeline(
        [
            ("minmax", payload["scalers"]["minmax"]),
            ("standard", payload["scalers"]["standard"]),
            ("clf", payload["model"]),
        ]
    )

    labels = payload["labels"]
    sample_label = "rice" if "rice" in labels else labels[0]
    sample = np.array([[90, 42, 43, 20.88, 82.0, 6.5, 202.9]], dtype=np.float32)
    batch = np.repeat(sample, 1000, axis=0)

    pkl_load = []
    onnx_load = []
    for _ in range(N_LOAD):
        t0 = time.perf_counter()
        _ = pickle.load(open(BEST_MODEL_PKL, "rb"))
        pkl_load.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        onnx_load.append((time.perf_counter() - t0) * 1000)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    for _ in range(N_WARMUP):
        pipeline.predict(sample)
        sess.run(None, {"float_input": sample})

    pkl_times = []
    onnx_times = []
    for _ in range(N_BENCH):
        t0 = time.perf_counter()
        pipeline.predict(sample)
        pkl_times.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        sess.run(None, {"float_input": sample})
        onnx_times.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    pkl_batch_pred = pipeline.predict(batch)
    pkl_batch_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    onnx_batch_pred = sess.run(None, {"float_input": batch})[0]
    onnx_batch_ms = (time.perf_counter() - t0) * 1000

    payload_out = {
        "benchmark_model": payload["model_name"],
        "load_mean_ms": {"pkl": float(np.mean(pkl_load)), "onnx": float(np.mean(onnx_load))},
        "single_mean_ms": {"pkl": float(np.mean(pkl_times)), "onnx": float(np.mean(onnx_times))},
        "single_std_ms": {"pkl": float(np.std(pkl_times)), "onnx": float(np.std(onnx_times))},
        "batch_1000_mean_ms": {"pkl": float(pkl_batch_ms), "onnx": float(onnx_batch_ms)},
        "speedup": {
            "load": float(np.mean(pkl_load) / np.mean(onnx_load)),
            "single": float(np.mean(pkl_times) / np.mean(onnx_times)),
            "batch_1000": float(pkl_batch_ms / onnx_batch_ms),
        },
        "prediction_agreement": bool(np.array_equal(pkl_batch_pred.astype(str), np.asarray(onnx_batch_pred).reshape(-1).astype(str))),
        "sample_prediction_pkl": str(pipeline.predict(sample)[0]),
        "sample_prediction_onnx": str(np.asarray(sess.run(None, {"float_input": sample})[0]).reshape(-1)[0]),
    }
    OUT_JSON.write_text(json.dumps(payload_out, indent=2))
    print(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
