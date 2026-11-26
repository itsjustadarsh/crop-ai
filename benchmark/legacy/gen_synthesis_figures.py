"""
Generates synthesis benchmark figures (v2) for paper/assets/.

Uses results collected across benchmark sessions:
  - Section 1–6:  RF, GB, Stacking, AdaBoost, SVM, MLP
  - Section 7–10: XGBoost, TabNet, Hybrid DL+RF, VotingClassifier(RF500+ET500)

Outputs:
  synthesis_table_v2.png   — 10-method heatmap table
  synthesis_accuracy_chart_v2.png — bar chart, before/after augmentation
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ASSETS = Path(__file__).parent.parent / "paper" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
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

# ── 10-method results (80/20 stratified split on respective dataset) ──────────
# Each entry: [acc_before, prec_before, rec_before, f1_before,
#              acc_after,  prec_after,  rec_after,  f1_after]
# "before" = raw 2200-sample split; "after" = augmented 6600-sample split
RESULTS = {
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
    "TabNet":                      [0.9659, 0.9663, 0.9659, 0.9658,
                                    0.9871, 0.9872, 0.9871, 0.9871],
    "Hybrid DL+RF":                [0.9705, 0.9708, 0.9705, 0.9704,
                                    0.9902, 0.9903, 0.9902, 0.9902],
    "VotingClassifier\n(RF+ET)":   [0.9864, 0.9865, 0.9864, 0.9864,
                                    0.9977, 0.9977, 0.9977, 0.9977],
}

METHODS = list(RESULTS.keys())
N = len(METHODS)

# ── Figure 1: Accuracy Bar Chart ──────────────────────────────────────────────
print("Generating synthesis_accuracy_chart_v2.png …")

acc_b = [RESULTS[m][0] for m in METHODS]
acc_a = [RESULTS[m][4] for m in METHODS]

x   = np.arange(N)
w   = 0.38
fig, ax = plt.subplots(figsize=(16, 7), facecolor=P["bg"])
ax.set_facecolor(P["panel"])
for sp in ax.spines.values():
    sp.set_edgecolor(P["border"])

bars_b = ax.bar(x - w/2, acc_b, w, color=P["blue"],  alpha=0.85, label="Before augmentation", zorder=3)
bars_a = ax.bar(x + w/2, acc_a, w, color=P["green"], alpha=0.85, label="After augmentation",  zorder=3)

# Annotate the winner
best_idx = acc_a.index(max(acc_a))
ax.bar(x[best_idx] + w/2, acc_a[best_idx], w,
       color=P["amber"], alpha=0.95, zorder=4)
ax.text(x[best_idx] + w/2, acc_a[best_idx] + 0.003,
        f"{acc_a[best_idx]*100:.2f}%",
        ha="center", va="bottom", fontsize=8.5, color=P["amber"], fontweight="bold")

ax.set_xlim(-0.6, N - 0.4)
ax.set_ylim(0.48, 1.04)
ax.set_xticks(x)
ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=9, color=P["text"])
ax.set_ylabel("Accuracy", color=P["text"], fontsize=10)
ax.set_title("10-Method Synthesis Benchmark — Accuracy Before / After Augmentation",
             color=P["text"], fontsize=12, fontweight="bold", pad=12)
ax.tick_params(colors=P["text"])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.grid(axis="y", color=P["border"], linewidth=0.5, zorder=0)

legend = ax.legend(facecolor=P["panel"], edgecolor=P["border"],
                   labelcolor=P["text"], fontsize=9)
patch_winner = mpatches.Patch(color=P["amber"], label="Best model (Voting RF+ET)")
ax.legend(handles=[bars_b, bars_a, patch_winner],
          labels=["Before augmentation", "After augmentation", "Best: VotingClassifier (99.77%)"],
          facecolor=P["panel"], edgecolor=P["border"], labelcolor=P["text"], fontsize=9)

plt.tight_layout()
out = ASSETS / "synthesis_accuracy_chart_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"  Saved {out}")

# ── Figure 2: Synthesis Table Heatmap ────────────────────────────────────────
print("Generating synthesis_table_v2.png …")

COL_LABELS = ["Acc (B)", "Prec (B)", "Rec (B)", "F1 (B)",
              "Acc (A)", "Prec (A)", "Rec (A)", "F1 (A)"]
data = np.array([RESULTS[m] for m in METHODS])

# Normalise each column 0→1 for colour mapping (skip AdaBoost from norm range)
mask_ada = np.array([m == "AdaBoost (SAMME)" for m in METHODS])
data_norm = np.zeros_like(data)
for j in range(data.shape[1]):
    col = data[:, j]
    non_ada = col[~mask_ada]
    cmin, cmax = non_ada.min(), non_ada.max()
    data_norm[:, j] = (col - cmin) / (cmax - cmin + 1e-9)
data_norm = np.clip(data_norm, 0, 1)

fig, ax = plt.subplots(figsize=(16, 8), facecolor=P["bg"])
ax.set_facecolor(P["panel"])
ax.set_xlim(-0.5, len(COL_LABELS) - 0.5)
ax.set_ylim(-0.5, N - 0.5)
ax.invert_yaxis()

cmap = plt.cm.RdYlGn
for i, method in enumerate(METHODS):
    for j, val in enumerate(data[i]):
        norm_val = data_norm[i, j]
        color = cmap(norm_val)
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                               facecolor=color, edgecolor=P["border"], linewidth=0.5)
        ax.add_patch(rect)
        text_color = "black" if norm_val > 0.5 else "white"
        ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold")

ax.set_xticks(range(len(COL_LABELS)))
ax.set_xticklabels(COL_LABELS, color=P["text"], fontsize=9.5)
ax.set_yticks(range(N))
clean_labels = [m.replace("\n", " ") for m in METHODS]
ax.set_yticklabels(clean_labels, color=P["text"], fontsize=9.5)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_edgecolor(P["border"])

ax.set_title("10-Method Synthesis Benchmark — Full Metrics (B=Before / A=After Augmentation)",
             color=P["text"], fontsize=11, fontweight="bold", pad=12)

# Highlight VotingClassifier row
best_row = len(METHODS) - 1
for j in range(len(COL_LABELS)):
    rect = plt.Rectangle((j - 0.5, best_row - 0.5), 1, 1,
                           facecolor="none",
                           edgecolor=P["amber"], linewidth=2.5)
    ax.add_patch(rect)

plt.tight_layout()
out = ASSETS / "synthesis_table_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=P["bg"])
plt.close()
print(f"  Saved {out}")

print("\nDone. Both synthesis figures written to paper/assets/")
