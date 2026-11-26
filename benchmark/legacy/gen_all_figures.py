"""
gen_all_figures.py — Generate all summary/architecture figures for paper/assets/.

Depends on artifacts saved by crop_benchmark_v2.py:
  paper/assets/RESULTS_12.pkl

Generates:
  synthesis_accuracy_chart_v3.png  — 11-method bar chart
  synthesis_table_v3.png           — 11-method full-metrics heatmap
  dl_architecture.png              — 1D CNN architecture diagram
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

BASE   = Path(__file__).parent.parent
ASSETS = BASE / "paper" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ── Dark theme palette ────────────────────────────────────────────────────────
P = {
    "bg":     "#0a0a0a",
    "panel":  "#111111",
    "border": "#1e1e1e",
    "text":   "#e0e0e0",
    "sub":    "#888888",
    "green":  "#22c55e",
    "blue":   "#3b82f6",
    "amber":  "#f59e0b",
    "red":    "#ef4444",
    "purple": "#a855f7",
    "teal":   "#14b8a6",
    "pink":   "#ec4899",
}

# ── Load RESULTS_12 ───────────────────────────────────────────────────────────
results_path = ASSETS / "RESULTS_12.pkl"
if results_path.exists():
    with open(results_path, "rb") as f:
        RESULTS_12 = pickle.load(f)
    print(f"Loaded RESULTS_12 ({len(RESULTS_12)} methods) from {results_path}")
else:
    # fallback: use hardcoded values (for standalone testing)
    print("RESULTS_12.pkl not found — using hardcoded fallback values")
    RESULTS_12 = {
        "Random Forest":               [0.9545, 0.9551, 0.9545, 0.9545,
                                        0.9955, 0.9955, 0.9955, 0.9955],
        "Gradient Boosting":           [0.9682, 0.9685, 0.9682, 0.9682,
                                        0.9886, 0.9887, 0.9886, 0.9886],
        "Stacking":                    [0.9591, 0.9596, 0.9591, 0.9590,
                                        0.9955, 0.9956, 0.9955, 0.9955],
        "AdaBoost (SAMME)":            [0.5500, 0.5423, 0.5500, 0.5380,
                                        0.6023, 0.5967, 0.6023, 0.5921],
        "SVM (RBF)":                   [0.9500, 0.9505, 0.9500, 0.9499,
                                        0.9818, 0.9821, 0.9818, 0.9818],
        "MLP (128,64)":                [0.9614, 0.9618, 0.9614, 0.9613,
                                        0.9932, 0.9932, 0.9932, 0.9932],
        "XGBoost":                     [0.9750, 0.9753, 0.9750, 0.9750,
                                        0.9932, 0.9932, 0.9932, 0.9932],
        "Hybrid DL+RF":                [0.9705, 0.9708, 0.9705, 0.9704,
                                        0.9902, 0.9903, 0.9902, 0.9902],
        "VotingClassifier\n(RF+ET)":   [0.9864, 0.9865, 0.9864, 0.9864,
                                        0.9977, 0.9977, 0.9977, 0.9977],
        "1D CNN":                      [0.9659, 0.9661, 0.9659, 0.9658,
                                        0.9932, 0.9932, 0.9932, 0.9932],
        "Deep Voting\nEnsemble":       [0.9886, 0.9887, 0.9886, 0.9886,
                                        0.9985, 0.9985, 0.9985, 0.9985],
    }

METHODS = list(RESULTS_12.keys())
N = len(METHODS)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: synthesis_accuracy_chart_v3.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating synthesis_accuracy_chart_v3.png ...")

acc_b = [RESULTS_12[m][0] for m in METHODS]
acc_a = [RESULTS_12[m][4] for m in METHODS]

best_idx = acc_a.index(max(acc_a))
best_acc = acc_a[best_idx]
best_name = METHODS[best_idx].replace("\n", " ")

x = np.arange(N)
w = 0.38

fig, ax = plt.subplots(figsize=(18, 7), facecolor=P["bg"])
ax.set_facecolor(P["panel"])
for sp in ax.spines.values():
    sp.set_edgecolor(P["border"])

bars_b = ax.bar(x - w/2, acc_b, w, color=P["blue"],  alpha=0.85,
                label="Before augmentation", zorder=3)
bars_a = ax.bar(x + w/2, acc_a, w, color=P["green"], alpha=0.85,
                label="After augmentation",  zorder=3)

# Highlight the best model (after augmentation)
ax.bar(x[best_idx] + w/2, acc_a[best_idx], w,
       color=P["amber"], alpha=0.95, zorder=4)
ax.text(x[best_idx] + w/2, acc_a[best_idx] + 0.004,
        f"{acc_a[best_idx]*100:.2f}%",
        ha="center", va="bottom", fontsize=8.5,
        color=P["amber"], fontweight="bold")

# Add accuracy labels on top of each after-aug bar
for i, (xi, v) in enumerate(zip(x, acc_a)):
    if i != best_idx and v > 0.85:
        ax.text(xi + w/2, v + 0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6, color=P["sub"])

ax.set_xlim(-0.6, N - 0.4)
ax.set_ylim(0.48, 1.06)
ax.set_xticks(x)
ax.set_xticklabels([m.replace("\n", "\n") for m in METHODS],
                   rotation=28, ha="right", fontsize=9, color=P["text"])
ax.set_ylabel("Accuracy", color=P["text"], fontsize=11)
ax.set_title("12-Method Synthesis Benchmark — Accuracy Before / After Augmentation",
             color=P["text"], fontsize=13, fontweight="bold", pad=14)
ax.tick_params(colors=P["text"])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="y", color=P["border"], linewidth=0.5, zorder=0)

patch_winner = mpatches.Patch(color=P["amber"],
                               label=f"Best: {best_name} ({best_acc*100:.2f}%)")
ax.legend(handles=[bars_b, bars_a, patch_winner],
          labels=["Before augmentation", "After augmentation",
                  f"Best: {best_name} ({best_acc*100:.2f}%)"],
          facecolor=P["panel"], edgecolor=P["border"],
          labelcolor=P["text"], fontsize=9)

plt.tight_layout()
out = ASSETS / "synthesis_accuracy_chart_v3.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"  Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: synthesis_table_v3.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating synthesis_table_v3.png ...")

COL_LABELS = ["Acc (B)", "Prec (B)", "Rec (B)", "F1 (B)",
              "Acc (A)", "Prec (A)", "Rec (A)", "F1 (A)"]
data = np.array([RESULTS_12[m] for m in METHODS])

# Normalise each column 0→1, skip AdaBoost rows from norm range
ada_mask = np.array(["AdaBoost" in m for m in METHODS])
data_norm = np.zeros_like(data)
for j in range(data.shape[1]):
    col = data[:, j]
    non_ada = col[~ada_mask]
    cmin, cmax = non_ada.min(), non_ada.max()
    data_norm[:, j] = (col - cmin) / (cmax - cmin + 1e-9)
data_norm = np.clip(data_norm, 0, 1)

fig, ax = plt.subplots(figsize=(18, 9), facecolor=P["bg"])
ax.set_facecolor(P["panel"])
ax.set_xlim(-0.5, len(COL_LABELS) - 0.5)
ax.set_ylim(-0.5, N - 0.5)
ax.invert_yaxis()

cmap = plt.cm.RdYlGn
for i, method in enumerate(METHODS):
    for j in range(len(COL_LABELS)):
        norm_val = data_norm[i, j]
        color = cmap(norm_val)
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                               facecolor=color, edgecolor=P["border"], linewidth=0.5)
        ax.add_patch(rect)
        tc = "black" if norm_val > 0.5 else "white"
        ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center",
                fontsize=8.5, color=tc, fontweight="bold")

ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, color=P["text"], fontsize=10)
ax.set_yticks(range(N))
ax.set_yticklabels([m.replace("\n", " ") for m in METHODS],
                   color=P["text"], fontsize=9.5)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_edgecolor(P["border"])

ax.set_title("12-Method Synthesis Benchmark — Full Metrics (B=Before / A=After Augmentation)",
             color=P["text"], fontsize=12, fontweight="bold", pad=14)

# Amber border on best row
for j in range(len(COL_LABELS)):
    rect = plt.Rectangle((j - 0.5, best_idx - 0.5), 1, 1,
                           facecolor="none", edgecolor=P["amber"], linewidth=2.5)
    ax.add_patch(rect)

plt.tight_layout()
out = ASSETS / "synthesis_table_v3.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"  Saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: dl_architecture.png
# ─────────────────────────────────────────────────────────────────────────────
print("Generating dl_architecture.png ...")

fig, ax = plt.subplots(figsize=(18, 6), facecolor=P["bg"])
ax.set_xlim(0, 18); ax.set_ylim(0, 6)
ax.set_facecolor(P["bg"])
ax.axis("off")
fig.suptitle("1D CNN Architecture for Crop Recommendation",
             color=P["text"], fontsize=13, fontweight="bold", y=0.97)

# Block definitions: (x_center, y_center, w, h, color, label, sublabel)
blocks = [
    (1.1,  3.0, 1.6, 1.8, P["teal"],   "Input\n(B × 7)",       "7 raw features"),
    (3.3,  3.0, 2.2, 1.8, P["blue"],   "Block 1\nConv1d(1→64)", "k=3,p=1 | BN | ReLU\nDrop(0.1)  →  (B,64,7)"),
    (5.9,  3.0, 2.2, 1.8, P["blue"],   "Block 2\nConv1d(64→128)", "k=3,p=1 | BN | ReLU\nDrop(0.1)  →  (B,128,7)"),
    (8.6,  3.0, 2.2, 1.8, P["purple"], "Block 3\nConv1d(128→64)", "k=3,p=1 | BN | ReLU\n(B,64,7)  ← GradCAM++"),
    (11.2, 3.0, 2.0, 1.8, P["teal"],   "GlobalAvgPool\n+ Squeeze", "(B,64)"),
    (13.3, 3.0, 2.0, 1.8, P["green"],  "FC Head\n(64→128→22)", "ReLU | Drop(0.3)\n→ logits (B,22)"),
    (15.7, 3.0, 1.6, 1.8, P["amber"],  "Output\nargmax+1", "crop label\n1..22"),
]

from matplotlib.patches import FancyBboxPatch

for bx, by, bw, bh, bc, label, sublabel in blocks:
    rect = FancyBboxPatch((bx - bw/2, by - bh/2), bw, bh,
                           boxstyle="round,pad=0.08",
                           facecolor=bc + "22", edgecolor=bc, linewidth=2)
    ax.add_patch(rect)
    ax.text(bx, by + 0.1, label, ha="center", va="center",
            fontsize=8.5, color=bc, fontweight="bold",
            multialignment="center")
    ax.text(bx, by - 0.52, sublabel, ha="center", va="center",
            fontsize=6.5, color=P["sub"], multialignment="center")

# GradCAM++ annotation box on Block 3
bx3, by3 = 8.6, 3.0
rect_gc = FancyBboxPatch((bx3 - 1.3, by3 - 1.05), 2.6, 2.1,
                          boxstyle="round,pad=0.05",
                          facecolor="none", edgecolor=P["red"],
                          linewidth=1.5, linestyle="--")
ax.add_patch(rect_gc)
ax.text(bx3, by3 + 1.28, "GradCAM++ hook",
        ha="center", va="center", fontsize=7.5,
        color=P["red"], fontweight="bold",
        bbox=dict(facecolor=P["bg"], edgecolor=P["red"], pad=2, linewidth=1))

# Arrows between blocks
arrow_xs = [1.9, 4.4, 7.0, 9.7, 12.2, 14.3]
arrow_targets = [3.3-1.1, 5.9-2.2, 8.6-2.2, 11.2-2.0, 13.3-2.0, 15.7-1.6]
for i, (ax_start_x, block_info) in enumerate(zip(arrow_xs, blocks[1:])):
    target_x = block_info[0] - block_info[2]/2
    mid_x = (ax_start_x + target_x) / 2 - 0.1
    ax.annotate("", xy=(target_x, 3.0), xytext=(ax_start_x, 3.0),
                arrowprops=dict(arrowstyle="->", color=P["sub"],
                               lw=1.5, mutation_scale=14))

# Legend
ax.text(9, 0.5, "BN = BatchNorm1d    |    ReLU = Rectified Linear Unit    |"
        "    Drop = Dropout    |    k = kernel_size    |    p = padding",
        ha="center", va="center", fontsize=7, color=P["sub"])

plt.tight_layout()
out = ASSETS / "dl_architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"  Saved {out}")

print("\n✅ gen_all_figures.py done. Figures:")
for p in sorted(ASSETS.glob("*.png")):
    print(f"  {p.name}")
