"""
PKL pipeline vs ONNX Runtime benchmark.
Measures: model load time, single inference time, batch inference time (1000 runs).
"""

import time
import pickle
import numpy as np
import onnxruntime as ort
from sklearn.pipeline import Pipeline

# Sample input: typical rice-growing conditions
SAMPLE = np.array([[90, 42, 43, 20.87, 82.0, 6.5, 202.9]], dtype=np.float32)
BATCH  = np.repeat(SAMPLE, 1000, axis=0)   # 1000 identical rows for batch test
N_WARMUP = 20
N_BENCH  = 1000

# ── PKL ──────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
model  = pickle.load(open("model.pkl",         "rb"))
minmax = pickle.load(open("minmaxscaler.pkl",   "rb"))
std    = pickle.load(open("standscaler.pkl",    "rb"))
pipeline = Pipeline([("minmax", minmax), ("standard", std), ("clf", model)])
pkl_load_ms = (time.perf_counter() - t0) * 1000

# warmup
for _ in range(N_WARMUP):
    pipeline.predict(SAMPLE)

# single inference
times = []
for _ in range(N_BENCH):
    t = time.perf_counter()
    pipeline.predict(SAMPLE)
    times.append((time.perf_counter() - t) * 1000)
pkl_single_ms  = np.mean(times)
pkl_single_std = np.std(times)

# batch (1000 samples at once)
t0 = time.perf_counter()
pipeline.predict(BATCH)
pkl_batch_ms = (time.perf_counter() - t0) * 1000

# ── ONNX ─────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
sess = ort.InferenceSession("../crop_model.onnx",
       providers=["CPUExecutionProvider"])
onnx_load_ms = (time.perf_counter() - t0) * 1000

tensor = SAMPLE  # ONNX pipeline includes scalers, takes raw input
batch_tensor = BATCH

# warmup
for _ in range(N_WARMUP):
    sess.run(None, {"float_input": tensor})

# single inference
times = []
for _ in range(N_BENCH):
    t = time.perf_counter()
    sess.run(None, {"float_input": tensor})
    times.append((time.perf_counter() - t) * 1000)
onnx_single_ms  = np.mean(times)
onnx_single_std = np.std(times)

# batch (1000 samples at once)
t0 = time.perf_counter()
sess.run(None, {"float_input": batch_tensor})
onnx_batch_ms = (time.perf_counter() - t0) * 1000

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Metric':<35} {'PKL':>10} {'ONNX':>10}  {'Speedup':>8}")
print("="*60)
print(f"{'Model load time (ms)':<35} {pkl_load_ms:>10.2f} {onnx_load_ms:>10.2f}  {pkl_load_ms/onnx_load_ms:>7.2f}x")
print(f"{'Single inference avg (ms)':<35} {pkl_single_ms:>10.4f} {onnx_single_ms:>10.4f}  {pkl_single_ms/onnx_single_ms:>7.2f}x")
print(f"{'Single inference std (ms)':<35} {pkl_single_std:>10.4f} {onnx_single_std:>10.4f}")
print(f"{'Batch/1000 inference (ms)':<35} {pkl_batch_ms:>10.2f} {onnx_batch_ms:>10.2f}  {pkl_batch_ms/onnx_batch_ms:>7.2f}x")
print("="*60)

# Verify same prediction
pkl_pred  = pipeline.predict(SAMPLE)[0]
onnx_pred = sess.run(None, {"float_input": tensor})[0][0]
print(f"\nPKL prediction : {pkl_pred}")
print(f"ONNX prediction: {onnx_pred}")
print(f"Agreement      : {'YES' if str(pkl_pred) == str(onnx_pred) else 'NO'}")
