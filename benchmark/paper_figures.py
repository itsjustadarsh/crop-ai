"""Generate every paper figure from REAL experiment outputs (no hardcoded values).

Inputs : paper/assets/onnx_winner_results.json   (benchmark/onnx_winner_benchmark.py)
         paper/assets/tuned_ensembles.json        (benchmark/tuned_ensembles.py)
         paper/assets/onnx_runtime_results.json   (benchmark/onnx_runtime_benchmark.py)
         model/research_best_model.pkl            (RF winner + scalers)
Outputs: paper/figs/*.pdf, paper/assets/xai_results.json
Usage  : python benchmark/paper_figures.py [all|workflow|accuracy|augment|confusion|xai|speed]
"""
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-crop-ai")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
ASSETS = BASE / "paper" / "assets"
FIGS = BASE / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
DATA = Path.home() / ".cache/kagglehub/datasets/atharvaingle/crop-recommendation-dataset/versions/1/Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
NICE = {"N": "N", "P": "P", "K": "K", "temperature": "Temperature", "humidity": "Humidity", "ph": "pH", "rainfall": "Rainfall"}

# One colourful, print-safe palette used everywhere
C = {"rf": "#2E7D32", "stack": "#7B1FA2", "et": "#00838F", "gb": "#EF6C00", "svm": "#1565C0",
     "mlp": "#C62828", "vote": "#AD1457", "grey": "#616161", "light": "#ECEFF1",
     "onnx": "#00897B", "pkl": "#546E7A", "pos": "#2E7D32", "neg": "#C62828"}
MODEL_COLOR = {"Random Forest": C["rf"], "Extra Trees": C["et"], "Gradient Boosting": C["gb"],
               "SVM (RBF)": C["svm"], "MLP": C["mlp"], "Stacking": C["stack"],
               "Stacking (LR meta, tuned)": C["stack"], "Stacking (RF meta, tuned)": "#9C27B0",
               "Soft Voting (tuned)": C["vote"], "Super-stack": "#4A148C", "CatBoost": "#F9A825", "XGBoost": "#5D4037",
               "LightGBM": "#1565C0", "k-NN (k=5)": "#78909C", "k-NN": "#78909C"}

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL_W, PAGE_W = 3.45, 7.1


def jload(name):
    return json.loads((ASSETS / name).read_text())


def jtry(name):
    f = ASSETS / name
    return json.loads(f.read_text()) if f.exists() else None


def pending(ax, title):
    ax.set_title(title, loc="left", fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.5, "pending", ha="center", va="center", color="#B71C1C", fontsize=8, transform=ax.transAxes)


def split():
    df = pd.read_csv(DATA)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def winner():
    p = pickle.loads((BASE / "model" / "research_best_model.pkl").read_bytes())
    mm, sd = p["scalers"]["minmax"], p["scalers"]["standard"]
    labels = np.array(p["labels"])
    scale = lambda X: sd.transform(mm.transform(X))  # noqa: E731
    return p["model"], scale, labels


# ---------------------------------------------------------------- Fig 1
def fig_workflow():
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def row(steps, y, h, w, gap, x0, labels=None):
        for i, (title, body, col) in enumerate(steps):
            x = x0 + i * (w + gap)
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.4",
                                        fc=col, ec="none", alpha=0.95))
            ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")
            ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center", color="white", fontsize=5.8, linespacing=1.12)
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + w + gap - 0.15, y + h / 2), xytext=(x + w + 0.35, y + h / 2),
                            arrowprops=dict(arrowstyle="-|>", color="#37474F", lw=1.1, mutation_scale=8))
                if labels:
                    ax.text(x + w + gap / 2, y + h / 2 - 1.6, labels[i], ha="center", va="top", fontsize=5.2,
                            color="#37474F", style="italic")

    ax.text(0.3, 38.6, "(a) Model study (offline)", fontsize=7.2, fontweight="bold", color="#263238", va="center")
    row([("Data", "2,200 × 7\n22 crops", "#455A64"),
         ("Split", "80/20 stratified\ntest locked", "#1565C0"),
         ("Inner CV", "10 families\n+ tuned stacks", "#00838F"),
         ("Select", "rank by\nCV accuracy", "#2E7D32"),
         ("Stress tests", "split luck · noise\n· input subsets", "#EF6C00"),
         ("Explain", "Gini · SHAP\n· LIME", "#AD1457"),
         ("Shortlist", "conformal\ncrop sets", "#6A1B9A")], y=21.5, h=14.5, w=12.4, gap=2.0, x0=0.5)

    ax.text(0.3, 17.6, "(b) Deployed pipeline (online; one HTTP endpoint per stage)", fontsize=7.2,
            fontweight="bold", color="#263238", va="center")
    row([("GPS point", "latitude,\nlongitude", "#37474F"),
         ("SoilGrids 2.0", "0–5 cm, 250 m\nN, pH; P, K proxies", "#6D4C41"),
         ("Open-Meteo", "temp., humidity,\n30-day rainfall", "#0277BD"),
         ("ONNX RF", "7 features → crop\n≈0.05 ms", "#00897B"),
         ("LLM advice", "Llama 3.3 70B\nplain-language plan", "#AD1457")],
        y=0.5, h=14.0, w=15.6, gap=5.0, x0=0.5,
        labels=["lat,\nlon", "N, P,\nK, pH", "+T, H,\nR", "crop"])
    fig.savefig(FIGS / "fig_workflow.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_accuracy():
    base = [r for r in jload("onnx_winner_results.json")["results"] if r["dataset_variant"] == "raw"]
    tuned = jload("tuned_ensembles.json")["results"] if (ASSETS / "tuned_ensembles.json").exists() else []
    push = jtry("accuracy_push_locked.json")
    if push and "Super-stack" in push["locked"]:
        ss = push["locked"]["Super-stack"]
        tuned = tuned + [{"model_name": "Super-stack", "cv_mean_accuracy": ss["cv_mean"], "cv_std_accuracy": ss["cv_std"],
                          "test_accuracy": ss["test_acc"]}]
    rows = sorted(base, key=lambda r: r["cv_mean_accuracy"]) + sorted(tuned, key=lambda r: r["cv_mean_accuracy"])
    names = [r["model_name"].replace(" (", "\n(") if len(r["model_name"]) > 14 else r["model_name"] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(COL_W, 0.27 * len(rows) + 0.55))
    for i, r in enumerate(rows):
        col = MODEL_COLOR.get(r["model_name"], C["grey"])
        ax.errorbar(r["cv_mean_accuracy"] * 100, i, xerr=r["cv_std_accuracy"] * 100, fmt="o", color=col,
                    ms=4.2, capsize=2, lw=1)
        ax.plot(r["test_accuracy"] * 100, i, marker="D", ms=4.2, mfc="white", mec=col, mew=1.3, ls="none")
        ax.text(100.38, i, f"{r['test_accuracy']*100:.2f}", va="center", fontsize=6.3, color=col)
    if tuned:
        ax.axhline(len(base) - 0.5, color="#B0BEC5", lw=0.7, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(97.0, 100.3); ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", color="#ECEFF1", lw=0.6)
    ax.plot([], [], "o", color=C["grey"], label="inner-CV mean ± std")
    ax.plot([], [], "D", mfc="white", mec=C["grey"], label="held-out test")
    ax.legend(loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(-0.02, 1.0))
    ax.text(100.38, len(rows) - 0.35, "test", fontsize=6, color=C["grey"], va="bottom")
    fig.savefig(FIGS / "fig_accuracy.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_augment():
    res = jload("onnx_winner_results.json")["results"]
    raw = {r["model_name"]: r for r in res if r["dataset_variant"] == "raw"}
    aug = {r["model_name"]: r for r in res if r["dataset_variant"] == "augmented"}
    order = sorted(raw, key=lambda n: -raw[n]["cv_mean_accuracy"])
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    rv = [raw[n]["test_accuracy"] * 100 for n in order]
    av = [aug[n]["test_accuracy"] * 100 for n in order]
    ax.bar(x - w / 2, rv, w, color="#90A4AE", label="raw training data")
    ax.bar(x + w / 2, av, w, color=[MODEL_COLOR[n] for n in order], label="+ Gaussian augmentation")
    for i in range(len(order)):
        d = av[i] - rv[i]
        col = C["pos"] if d > 0 else (C["neg"] if d < 0 else C["grey"])
        ax.text(x[i], max(rv[i], av[i]) + 0.06, f"{d:+.2f}", ha="center", fontsize=6.4, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (RBF)", "").replace("Gradient Boosting", "Grad.\nBoosting")
                                          .replace("Random Forest", "Random\nForest").replace("Extra Trees", "Extra\nTrees") for n in order], fontsize=6.4)
    ax.set_ylim(98.0, 100.25); ax.set_ylabel("Test accuracy (%)")
    ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.17))
    ax.grid(axis="y", color="#ECEFF1", lw=0.6); ax.set_axisbelow(True)
    fig.savefig(FIGS / "fig_augment.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_confusion():
    _, Xte, _, yte = split()
    model, scale, labels = winner()
    pred = labels[model.predict(scale(Xte))]
    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(COL_W, 3.1))
    off = cm.copy(); np.fill_diagonal(off, 0)
    ax.imshow(np.where(np.eye(len(labels)) == 1, cm, np.nan), cmap="Greens", vmin=0, vmax=22)
    ax.imshow(np.where(off > 0, off, np.nan), cmap="Reds", vmin=0, vmax=2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=4.6,
                        color="white" if i == j else "black", fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5.2); ax.set_yticklabels(labels, fontsize=5.2)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.savefig(FIGS / "fig_confusion.pdf"); plt.close(fig)
    errors = [{"true": str(yte[i]), "pred": str(pred[i])} for i in np.where(pred != yte)[0]]
    return {"test_correct": int((pred == yte).sum()), "test_n": int(len(yte)), "errors": errors}


# ---------------------------------------------------------------- Fig 5
def fig_xai():
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    Xtr, Xte, ytr, yte = split()
    model, scale, labels = winner()
    rice = int(np.where(labels == "rice")[0][0])
    gini = dict(zip(FEATURES, model.feature_importances_.tolist()))

    Xte_s = scale(Xte)
    sv = shap.TreeExplainer(model).shap_values(Xte_s)          # (n, f, classes) or list
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    shap_global = dict(zip(FEATURES, np.abs(sv).mean(axis=(0, 2)).tolist()))
    rice_rows = np.where(yte == "rice")[0]
    shap_rice = dict(zip(FEATURES, np.abs(sv[rice_rows, :, rice]).mean(axis=0).tolist()))

    pred = labels[model.predict(Xte_s)]
    idx = int(next(i for i in rice_rows if pred[i] == "rice"))
    proba = lambda Xraw: model.predict_proba(scale(np.asarray(Xraw, dtype=np.float32)))  # noqa: E731
    lime_exp = LimeTabularExplainer(Xtr, feature_names=[NICE[f] for f in FEATURES], class_names=list(labels),
                                    discretize_continuous=True, random_state=42)
    exp = lime_exp.explain_instance(Xte[idx], proba, labels=(rice,), num_features=7, num_samples=5000)
    lime_pairs = exp.as_list(label=rice)

    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.75), gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 1.05})
    feat_cols = ["#5D4037", "#F9A825", "#8E24AA", "#E53935", "#1E88E5", "#43A047", "#00ACC1"]
    fc = dict(zip(FEATURES, feat_cols))
    order = sorted(gini, key=gini.get)
    axes[0].barh([NICE[f] for f in order], [gini[f] for f in order], color=[fc[f] for f in order])
    axes[0].set_title("(a) RF impurity importance", loc="left", fontweight="bold")
    axes[0].set_xlabel("mean decrease in impurity")
    # (b) SHAP, each bar normalised to its own total so global and rice are comparable
    g = np.array([shap_global[f] for f in order]); g = g / g.sum()
    rc = np.array([shap_rice[f] for f in order]); rc = rc / rc.sum()
    yy = np.arange(len(order)); h = 0.4
    axes[1].barh(yy + h / 2, g, h, color="#90A4AE", label="all 22 classes")
    axes[1].barh(yy - h / 2, rc, h, color=[fc[f] for f in order], label="rice output")
    axes[1].set_yticks(yy); axes[1].set_yticklabels([NICE[f] for f in order])
    axes[1].set_title("(b) SHAP share of attribution", loc="left", fontweight="bold")
    axes[1].set_xlabel("share of mean |SHAP|"); axes[1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.08, -0.02), fontsize=5.6, handlelength=1)
    for ax in axes[:2]:
        ax.grid(axis="x", color="#ECEFF1", lw=0.6); ax.set_axisbelow(True)
    names = [n for n, _ in lime_pairs][::-1]; vals = [v for _, v in lime_pairs][::-1]
    axes[2].barh(names, vals, color=[C["pos"] if v > 0 else C["neg"] for v in vals])
    axes[2].axvline(0, color="#37474F", lw=0.7)
    axes[2].set_title("(c) LIME, one rice test sample", loc="left", fontweight="bold")
    axes[2].set_xlabel("weight toward P(rice)"); axes[2].tick_params(axis="y", labelsize=6)
    fig.savefig(FIGS / "fig_xai.pdf"); plt.close(fig)
    rank = lambda d: sorted(d, key=d.get, reverse=True)  # noqa: E731
    return {"gini": gini, "gini_rank": rank(gini), "shap_global": shap_global, "shap_global_rank": rank(shap_global),
            "shap_rice": shap_rice, "shap_rice_rank": rank(shap_rice), "lime_rice_sample_index": idx,
            "lime_rice_sample_raw": dict(zip(FEATURES, Xte[idx].tolist())),
            "lime_rice_prob": float(proba(Xte[idx:idx + 1])[0, rice]), "lime": lime_pairs,
            "lime_settings": {"num_samples": 5000, "discretize_continuous": True, "random_state": 42}}


# ---------------------------------------------------------------- Fig 6
def fig_speed():
    r = jload("onnx_runtime_results.json")
    cats = ["Model load", "Single prediction", "Batch of 1,000"]
    pkl = [r["load_ms"]["pkl"][0], r["single_ms"]["pkl"][0], r["batch_1000_ms"]["pkl"][0]]
    onx = [r["load_ms"]["onnx"][0], r["single_ms"]["onnx"][0], r["batch_1000_ms"]["onnx"][0]]
    stk = [np.nan, r["stacking_sklearn"]["single_ms"][0], r["stacking_sklearn"]["batch_1000_ms"][0]]
    x = np.arange(3); w = 0.27
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    b1 = ax.bar(x - w, stk, w, color=C["stack"], label="Stacking (scikit-learn)")
    b2 = ax.bar(x, pkl, w, color=C["pkl"], label="RF (scikit-learn)")
    b3 = ax.bar(x + w, onx, w, color=C["onnx"], label="RF (ONNX Runtime)")
    ax.set_yscale("log"); ax.set_ylabel("Time (ms, log scale)")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h * 1.15, f"{h:.3g}", ha="center", fontsize=5.8)
    for i, k in enumerate(["load", "single", "batch_1000"]):
        s = r["speedup"][k]
        ax.text(x[i] + w / 2, ax.get_ylim()[1] if False else max(pkl[i], onx[i]) * 4.2,
                f"{s:.1f}×" if s >= 1 else f"{1/s:.1f}× slower", ha="center", fontsize=6.6, fontweight="bold",
                color=C["onnx"] if s >= 1 else C["neg"])
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(top=ax.get_ylim()[1] * 12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=6.2, handlelength=1)
    ax.grid(axis="y", color="#ECEFF1", lw=0.6); ax.set_axisbelow(True)
    fig.savefig(FIGS / "fig_speed.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 7
def fig_repeated():
    """Repeated 5x10 CV: total errors per model, split into rice/jute vs all other crops."""
    rep = jload("accuracy_push_repeated.json")["repeated"]
    y = pd.read_csv(DATA)["label"].to_numpy()
    rows = []
    for m, r in rep.items():
        rj = sum(v for i, v in r["per_sample_errors"].items() if y[int(i)] in ("rice", "jute"))
        rows.append((m, r["mean"] * 100, r["std"] * 100, rj, r["errors_total"] - rj))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(COL_W, 0.27 * len(rows) + 0.6))
    yy = np.arange(len(rows))
    ax.barh(yy, [r[3] for r in rows], color="#00897B", label="rice / jute samples")
    ax.barh(yy, [r[4] for r in rows], left=[r[3] for r in rows], color="#B0BEC5", label="all other crops")
    for i, r in enumerate(rows):
        ax.text(r[3] + r[4] + 3, i, f"{r[1]:.2f}% ± {r[2]:.2f}", va="center", fontsize=6.3,
                color=MODEL_COLOR.get(r[0], C["grey"]) if r[0] in MODEL_COLOR else "#37474F", fontweight="bold")
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlabel("misclassifications in 11,000 out-of-fold predictions")
    ax.set_xlim(0, max(r[3] + r[4] for r in rows) * 1.42)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.grid(axis="x", color="#ECEFF1", lw=0.6); ax.set_axisbelow(True)
    fig.savefig(FIGS / "fig_repeated.pdf"); plt.close(fig)


# ---------------------------------------------------------------- data audit
def fig_audit():
    df = pd.read_csv(DATA)
    syn = jload("novelty_synthetic.json")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw={"wspace": 0.42, "width_ratios": [1.15, 1]})
    ax = axes[0]
    crops = sorted(df.label.unique()); cmap = plt.cm.tab20(np.linspace(0, 1, len(crops)))
    for c, col in zip(crops, cmap):
        d = df[df.label == c]
        ax.scatter(d.temperature, d.humidity, s=1.2, color=col, alpha=0.8, lw=0)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Humidity (%)")
    ax.set_title("(a) each crop fills a box", loc="left", fontweight="bold", fontsize=7)
    ax = axes[1]
    bins = np.linspace(150, 300, 26)
    ax.hist(df[df.label == "jute"].rainfall, bins=bins, color="#EF6C00", alpha=0.75, label="jute")
    ax.hist(df[df.label == "rice"].rainfall, bins=bins, color="#00897B", alpha=0.75, label="rice")
    ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("samples")
    ax.set_title("(b) flat, overlapping ranges", loc="left", fontweight="bold", fontsize=7)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    fig.savefig(FIGS / "fig_audit.pdf"); plt.close(fig)


# ---------------------------------------------------------------- beyond accuracy (4 panels)
def fig_beyond():
    inf = jtry("novelty_inflation.json"); con = jload("novelty_conformal.json")
    sub = jload("novelty_subsets.json"); noi = jload("novelty_noise.json")
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 1.7), gridspec_kw={"wspace": 0.5, "width_ratios": [1.05, 1.2, 1.0, 1.05]})
    # (a) inflation
    ax = axes[0]
    if inf is None:
        pending(ax, "(a) split luck")
    else:
        h = np.round(np.array(inf["honest_rf"]["accs"]) * 440).astype(int)
        b = np.round(np.array(inf["test_selected_pool"]["accs"]) * 440).astype(int)
        ks = np.arange(min(h.min(), b.min()), 441)
        ax.bar(ks - 0.2, [np.sum(h == k) for k in ks], 0.4, color="#2E7D32", label="RF, honest")
        ax.bar(ks + 0.2, [np.sum(b == k) for k in ks], 0.4, color="#C62828", label="best of 13, picked on test")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k}" for k in ks], fontsize=5.2, rotation=45)
        ax.set_xlabel("correct out of 440 test samples"); ax.set_ylabel("splits (of 100)")
        ax.set_title("(a) split luck", loc="left", fontweight="bold")
        ax.legend(frameon=False, fontsize=5.0, loc="upper left", handlelength=0.8, bbox_to_anchor=(-0.02, 1.02))
        ax.set_ylim(0, max(max(np.sum(h == k) for k in ks), max(np.sum(b == k) for k in ks)) * 1.45)
    # (b) conformal
    ax = axes[1]
    al = sorted(con["by_alpha"], key=float, reverse=True)
    xs = np.arange(len(al))
    nom = [100 * (1 - float(a)) for a in al]; cov = [100 * con["by_alpha"][a]["cov"] for a in al]
    size = [con["by_alpha"][a]["size"] for a in al]
    ax.plot(xs, nom, "_", color="#455A64", ms=11, mew=1.6, label="target")
    ax.plot(xs, cov, "o", color="#7B1FA2", ms=4, label="achieved")
    for x_, c_, s_ in zip(xs, cov, size):
        ax.text(x_, 100.2, f"{s_:.2f}", ha="center", fontsize=5.0, color="#4A148C")
    ax.text(-0.75, 100.2, "crops:", ha="right", fontsize=5.0, color="#4A148C", style="italic")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n:g}" for n in nom], fontsize=5.8)
    ax.set_ylim(94.5, 100.6); ax.set_xlim(-0.7, len(al) - 0.4)
    ax.set_xlabel("target coverage (%)"); ax.set_ylabel("coverage (%)")
    ax.set_title("(b) crop shortlists", loc="left", fontweight="bold", y=1.06)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1)
    # (c) feature subsets
    ax = axes[2]
    names = sorted(sub, key=lambda k: sub[k]["mean"])
    vals = [sub[k]["mean"] * 100 for k in names]
    cols = ["#1E88E5" if "Climate" in k and "N, P, K" not in k else "#8D6E63" if "Soil" in k else "#2E7D32" for k in names]
    SHORT = {"All seven": "all 7", "Climate + N, P, K": "climate+NPK", "All except rainfall": "all but rain",
             "Climate + pH": "climate+pH", "Climate (T, H, R)": "climate", "Soil (N, P, K, pH)": "soil only"}
    ax.barh([SHORT.get(n, n) for n in names], vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=5.6)
    ax.set_xlim(min(vals) - 8, 104); ax.tick_params(axis="y", labelsize=5.8)
    ax.set_xlabel("5×10 CV accuracy (%)")
    ax.set_title("(c) which inputs", loc="left", fontweight="bold")
    # (d) noise
    ax = axes[3]
    lv = [100 * l for l in noi["levels"]]
    show = ["Extra Trees", "Stacking", "Random Forest", "CatBoost", "XGBoost", "LightGBM"]
    for m, curve in ((m, noi["models"][m]) for m in show):
        ax.plot(lv, [100 * c for c in curve], "o-", ms=2.2, lw=1.0, color=MODEL_COLOR.get(m, C["grey"]),
                label=m.replace(" (RBF)", ""))
    ax.set_xlabel("input noise σ (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("(d) noisy inputs", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=4.8, ncol=1, loc="lower left", handlelength=1, labelspacing=0.15,
              bbox_to_anchor=(-0.02, -0.03))
    fig.savefig(FIGS / "fig_beyond.pdf"); plt.close(fig)


# ---------------------------------------------------------------- speed with Pareto
def fig_speed2():
    r = jload("onnx_runtime_results.json"); pa = jtry("novelty_pareto_acc.json"); ps = jtry("novelty_pareto_speed.json")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 1.75), gridspec_kw={"wspace": 0.28, "width_ratios": [1.15, 1]})
    ax = axes[0]
    cats = ["Model load", "Single prediction", "Batch of 1,000"]
    pkl = [r["load_ms"]["pkl"][0], r["single_ms"]["pkl"][0], r["batch_1000_ms"]["pkl"][0]]
    onx = [r["load_ms"]["onnx"][0], r["single_ms"]["onnx"][0], r["batch_1000_ms"]["onnx"][0]]
    stk = [np.nan, r["stacking_sklearn"]["single_ms"][0], r["stacking_sklearn"]["batch_1000_ms"][0]]
    x = np.arange(3); w = 0.27
    for off, vals, col, lab in [(-w, stk, C["stack"], "Stacking (scikit-learn)"), (0, pkl, C["pkl"], "RF-700 (scikit-learn)"),
                                (w, onx, C["onnx"], "RF-700 (ONNX Runtime)")]:
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for bb in bars:
            hh = bb.get_height()
            if np.isfinite(hh):
                ax.text(bb.get_x() + bb.get_width() / 2, hh * 1.15, f"{hh:.3g}", ha="center", fontsize=5.6)
    for i, k in enumerate(["load", "single", "batch_1000"]):
        s_ = r["speedup"][k]
        ax.text(x[i] + w / 2, max(pkl[i], onx[i]) * 4.5, f"{s_:.0f}×" if s_ >= 10 else (f"{s_:.1f}×" if s_ >= 1 else f"{1/s_:.1f}× slower"),
                ha="center", fontsize=6.4, fontweight="bold", color=C["onnx"] if s_ >= 1 else C["neg"])
    ax.set_yscale("log"); ax.set_ylabel("time (ms, log)"); ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(top=ax.get_ylim()[1] * 15)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=5.8, handlelength=1)
    ax.set_title("(a) RF-700 runtime", loc="left", fontweight="bold", y=1.13)
    ax = axes[1]
    if pa is None or ps is None:
        pending(ax, "(b) accuracy vs latency by number of trees")
        fig.savefig(FIGS / "fig_speed.pdf"); plt.close(fig); return
    trees = sorted(ps, key=int)
    acc = [pa[t]["mean"] * 100 for t in trees]
    ax.plot([ps[t]["sklearn_ms"] for t in trees], acc, "s--", color=C["pkl"], ms=3, lw=1, label="scikit-learn")
    ax.plot([ps[t]["onnx_ms"] for t in trees], acc, "o-", color=C["onnx"], ms=3.4, lw=1.2, label="ONNX Runtime")
    for t, a_ in zip(trees, acc):
        ax.annotate(t, (ps[t]["onnx_ms"], a_), textcoords="offset points", xytext=(3, -7), fontsize=5.4, color=C["onnx"])
    ax.set_xscale("log"); ax.set_xlabel("single-prediction latency (ms, log)"); ax.set_ylabel("5×10 CV accuracy (%)")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.set_title("(b) accuracy vs latency by number of trees", loc="left", fontweight="bold", y=1.13)
    for a in axes:
        a.grid(color="#ECEFF1", lw=0.6); a.set_axisbelow(True)
    fig.savefig(FIGS / "fig_speed.pdf"); plt.close(fig)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "repeated"): fig_repeated()
    if what in ("all", "audit"): fig_audit()
    if what in ("all", "beyond"): fig_beyond()
    if what in ("all", "speed2"): fig_speed2()
    out_path = ASSETS / "xai_results.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if what in ("all", "workflow"): fig_workflow()
    if what in ("all", "accuracy"): fig_accuracy()
    if what in ("all", "augment"): fig_augment()
    if what in ("all", "confusion"): out["confusion"] = fig_confusion()
    if what in ("all", "xai"): out.update(fig_xai())
    if what in ("all", "speed"): fig_speed()
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k in ("confusion", "gini_rank", "shap_global_rank", "shap_rice_rank", "lime")}, indent=1))


if __name__ == "__main__":
    main()
