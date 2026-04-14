"""
train_best_model.py — Load trained 1D CNN and export to ONNX
with embedded MinMax + Standard preprocessing.

Interface preserved (matches existing server.js):
  Input:  float_input   float32[1, 7]  (raw feature values)
  Output: label         int64[1]       (crop label 1..22)

Prerequisites (run crop_benchmark_v2.py first):
  model/cnn_model.pt
  model/scalers.pkl
"""
import pickle, json, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE      = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent

for fname in ["cnn_model.pt", "scalers.pkl"]:
    if not (MODEL_DIR / fname).exists():
        print(f"ERROR: {MODEL_DIR/fname} not found. Run crop_benchmark_v2.py first.")
        sys.exit(1)

# ── Load scaler params ────────────────────────────────────────────────────────
with open(MODEL_DIR / "scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
mm  = scalers["mm"]
std = scalers["std"]
mm_min    = mm.data_min_.astype(np.float32)
mm_max    = mm.data_max_.astype(np.float32)
std_mean  = std.mean_.astype(np.float32)
std_scale = std.scale_.astype(np.float32)
print("Scaler params loaded from model/scalers.pkl")

# ── CNN definition (must match crop_benchmark_v2.py) ──────────────────────────
class CropCNN(nn.Module):
    def __init__(self, n_classes=22):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1))
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 22))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        return self.fc(self.pool(x).squeeze(-1))

# ── Wrapper with embedded preprocessing ──────────────────────────────────────
class CropCNNWithPreprocessing(nn.Module):
    """Takes raw float32[B,7], returns int64 crop label [B] (1..22)."""
    def __init__(self, cnn, mm_min, mm_max, std_mean, std_scale):
        super().__init__()
        self.cnn = cnn
        self.register_buffer("mm_min",    torch.tensor(mm_min))
        self.register_buffer("mm_max",    torch.tensor(mm_max))
        self.register_buffer("std_mean",  torch.tensor(std_mean))
        self.register_buffer("std_scale", torch.tensor(std_scale))

    def forward(self, x):                                               # (B,7) raw
        x = (x - self.mm_min) / (self.mm_max - self.mm_min + 1e-8)    # MinMax
        x = (x - self.std_mean) / self.std_scale                        # Standard
        return self.cnn(x).argmax(dim=1) + 1                            # → 1..22

# ── Load weights ──────────────────────────────────────────────────────────────
cnn = CropCNN(22)
cnn.load_state_dict(torch.load(MODEL_DIR / "cnn_model.pt",
                                map_location="cpu", weights_only=True))
cnn.eval()
print("CNN weights loaded from model/cnn_model.pt")

wrapped = CropCNNWithPreprocessing(cnn, mm_min, mm_max, std_mean, std_scale)
wrapped.eval()

# ── Sanity check ──────────────────────────────────────────────────────────────
CROP_MAP = {
    1:"rice",2:"maize",3:"jute",4:"cotton",5:"coconut",6:"papaya",7:"orange",
    8:"apple",9:"muskmelon",10:"watermelon",11:"grapes",12:"mango",13:"banana",
    14:"pomegranate",15:"lentil",16:"blackgram",17:"mungbean",18:"mothbeans",
    19:"pigeonpeas",20:"kidneybeans",21:"chickpea",22:"coffee"
}
TEST_CASES = [
    ([90, 42, 43, 20.88, 82.0, 6.5,  202.9], "rice"),
    ([0,  67, 20, 22.0,  86.0, 6.4,  281.0], "banana"),
    ([29, 40, 70, 21.0,  90.0, 5.8,  209.0], "coconut"),
]
print("\nSanity check (pre-ONNX):")
for feat, exp_name in TEST_CASES:
    with torch.no_grad():
        pred_id = wrapped(torch.tensor([feat], dtype=torch.float32)).item()
    pred_name = CROP_MAP.get(pred_id, "?")
    status = "✅" if pred_name == exp_name else f"⚠️  got {pred_name}"
    print(f"  expected {exp_name:<12} → {pred_name:<12} {status}")

# ── Export to ONNX ────────────────────────────────────────────────────────────
out_onnx = BASE / "crop_model.onnx"
torch.onnx.export(
    wrapped,
    torch.zeros(1, 7, dtype=torch.float32),
    str(out_onnx),
    input_names=["float_input"],
    output_names=["label"],
    dynamic_axes={"float_input": {0: "batch_size"}, "label": {0: "batch_size"}},
    opset_version=17,
    do_constant_folding=True,
)
print(f"\n✅ ONNX exported → {out_onnx}")

# ── Verify with onnxruntime ───────────────────────────────────────────────────
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_onnx))
    print("ONNX Runtime verification:")
    for feat, exp_name in TEST_CASES:
        inp      = np.array([feat], dtype=np.float32)
        pred_id  = int(sess.run(None, {"float_input": inp})[0][0])
        pred_name = CROP_MAP.get(pred_id, "?")
        status = "✅" if pred_name == exp_name else f"⚠️  got {pred_name}"
        print(f"  expected {exp_name:<12} → {pred_name:<12} {status}")
except ImportError:
    print("onnxruntime not installed — skip verification")

# ── Update model_config.json ──────────────────────────────────────────────────
config_path = MODEL_DIR / "model_config.json"
cfg = json.loads(config_path.read_text()) if config_path.exists() else {}

results_path = BASE / "paper" / "assets" / "RESULTS_12.pkl"
if results_path.exists():
    with open(results_path, "rb") as f:
        R = pickle.load(f)
    cnn_acc = R.get("1D CNN", [None]*8)[4]
    dv_key  = next((k for k in R if "Deep Voting" in k), None)
    dv_acc  = R[dv_key][4] if dv_key else None
    best_acc = max(v[4] for v in R.values())
else:
    cnn_acc = dv_acc = best_acc = None

cfg.update({
    "model_type": "1D CNN (PyTorch) with embedded MinMax+Standard scaling — ONNX",
    "notes": "ONNX = 1D CNN. Research best = Deep Voting Ensemble (RF500+ET500+XGB300+CNN).",
    "scaler_params": {
        "minmax_feature_range": [0, 1],
        "minmax_data_min":  mm_min.tolist(),
        "minmax_data_max":  mm_max.tolist(),
        "standard_mean":    std_mean.tolist(),
        "standard_scale":   std_scale.tolist(),
    },
})
if cnn_acc: cfg["cnn_accuracy"]          = round(float(cnn_acc), 6)
if dv_acc:  cfg["deep_voting_accuracy"]  = round(float(dv_acc),  6)
if best_acc: cfg["model_accuracy"]       = round(float(best_acc), 6)

config_path.write_text(json.dumps(cfg, indent=2))
print(f"\n✅ model_config.json updated")
if cnn_acc:  print(f"   CNN accuracy:         {cnn_acc:.4f}")
if dv_acc:   print(f"   Deep Voting accuracy: {dv_acc:.4f}")
