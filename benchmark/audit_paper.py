"""Audit paper/paper.tex against every experiment output and the user's structural rules.
Prints PASS/FAIL per check; exit code 1 if anything fails."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
A = BASE / "paper" / "assets"
tex = (BASE / "paper" / "paper.tex").read_text()
bib = (BASE / "paper" / "references.bib").read_text()
J = lambda n: json.loads((A / n).read_text())  # noqa: E731
W, T, R, X = J("onnx_winner_results.json"), J("tuned_ensembles.json"), J("onnx_runtime_results.json"), J("xai_results.json")
P, RP, CE = J("accuracy_push_locked.json"), J("accuracy_push_repeated.json")["repeated"], J("ceiling.json")
SY, INF, CO, SU, NO, SG, PA, PS = (J(f"novelty_{k}.json") for k in
                                   ["synthetic", "inflation", "conformal", "subsets", "noise", "significance", "pareto_acc", "pareto_speed"])
LIT = J("literature.json")
y = pd.read_csv(Path.home() / ".cache/kagglehub/datasets/atharvaingle/crop-recommendation-dataset/versions/1/Crop_recommendation.csv")["label"].to_numpy()
lines = tex.splitlines()
fails = 0


def check(label, ok):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + label)


has = lambda s: s in tex  # noqa: E731
pct = lambda v: f"{v*100:.2f}"  # noqa: E731
raw = {r["model_name"]: r for r in W["results"] if r["dataset_variant"] == "raw"}
aug = {r["model_name"]: r for r in W["results"] if r["dataset_variant"] == "augmented"}
tab2 = [l for l in lines if l.count("&") == 7]   # rows of the two-block Table II


def in_tab2(left, exp):
    for l in tab2:
        cells = [c.strip() for c in l.replace("\\\\", "").split("&")]
        block = cells[:4] if left else cells[4:]
        if all(any(e in c for c in block) for e in exp):
            return True
    return False


# ---- Table II
for name in ["Random Forest", "Extra Trees", "Stacking", "Gradient Boosting", "MLP", "SVM (RBF)"]:
    r = raw[name]; exp = [pct(r["cv_mean_accuracy"]), pct(r["test_accuracy"]), str(round(r["test_accuracy"] * 440))]
    check(f"Table II {name} {exp}", in_tab2(True, exp))
for key in ["LightGBM (tuned)", "CatBoost (tuned)", "XGBoost (tuned)", "k-NN (tuned)", "Super-stack"]:
    r = P["locked"][key]; exp = [pct(r["cv_mean"]), pct(r["test_acc"]), str(r["test_correct"])]
    check(f"Table II {key} {exp}", in_tab2(False, exp))
t0 = T["results"][0]
check("Table II Stacking-LR tuned", in_tab2(False, ["Stacking-LR", pct(t0["cv_mean_accuracy"]), str(t0["test_correct"])]))
check("super-stack base learners", P["superstack_selection"]["chosen"] == ["LightGBM", "Random Forest", "Extra Trees", "XGBoost", "CatBoost"])
check("feature engineering rejected", not P["feature_engineering"]["adopted"] and has("rejected"))
gb = round((aug["Gradient Boosting"]["test_accuracy"] - raw["Gradient Boosting"]["test_accuracy"]) * 440)
check(f"GB augmentation +{gb} samples", has(f"(+{gb} samples)"))

# ---- repeated CV, significance, ceiling
for name, txt in [("Random Forest", "99.53"), ("Stacking", "99.54"), ("Extra Trees", "99.38"), ("CatBoost", "99.35"),
                  ("XGBoost", "99.34"), ("LightGBM", "99.16")]:
    check(f"repeated CV {name} {txt}", pct(RP[name]["mean"]) == txt and has(txt + "\\%"))
check("errors 51 / 52", RP["Stacking"]["errors_total"] == 51 and RP["Random Forest"]["errors_total"] == 52
      and has("51 errors") and has("52 errors"))
check("significance re-run reproduces means", all(abs(SG["means"][k] - RP[k]["mean"]) < 1e-9 for k in SG["means"]))
check("no significant pair; text says tied", all(v["p"] >= 0.05 for v in SG["pairs"].values()) and has("statistically tied"))
p_sr = SG["pairs"]["Random Forest vs Stacking"]["p"]
check(f"stacking vs RF p={p_sr:.2f}", has(f"$p={p_sr:.2f}$"))
sh = lambda m: sum(v for i, v in RP[m]["per_sample_errors"].items() if y[int(i)] in ("rice", "jute")) / RP[m]["errors_total"]  # noqa: E731
check("rice/jute share 77% / 85%", f"{sh('Random Forest')*100:.0f}" == "77" and f"{sh('Extra Trees')*100:.0f}" == "85"
      and has("77\\% of RF's and 85\\% of ET's"))
hard = CE["samples_wrong_majority_every_repeat"]
check("9 hard samples, 8 rice/jute, none wrong for all", len(hard) == 9 and sum(y[i] in ("rice", "jute") for i in hard) == 8
      and CE["samples_wrong_every_model_every_repeat"] == [] and has("Nine") and has("eight are rice"))
check("super-stack's error is lentil", P["locked"]["Super-stack"]["test_errors"] == [["lentil", "mothbeans"]] and has("misses a lentil"))

# ---- data audit
check(f"audit {SY['share_pairs_significant_p05']*100:.1f}% of {SY['n_pairs']} pairs, mean |r| {SY['within_class_mean_abs_r']:.2f}",
      has(f"{SY['share_pairs_significant_p05']*100:.1f}\\%") and has(str(SY["n_pairs"])) and has(f"{SY['within_class_mean_abs_r']:.2f}"))
check("audit 141/154 uniform, normal rejected for all", has(f"In {SY['per_class_feature_uniform_not_rejected']} of the")
      and has(str(SY["per_class_feature_total"])) and SY["per_class_feature_normal_not_rejected"] == 0)
check("audit P-K r, no duplicates", has(f"r={SY['global_PK_r']:.2f}") and SY["exact_duplicate_rows"] == 0)

# ---- inflation
h, pool = INF["honest_rf"], INF["test_selected_pool"]
check(f"inflation range {pct(h['min'])}-{pct(h['max'])}, mean {pct(h['mean'])}",
      has(f"from {pct(h['min'])} to {pct(h['max'])}") and has(f"mean {pct(h['mean'])}"))
check("10% of splits score 100%", f"{h['share_100']*100:.0f}" == "10" and has("10\\% of splits score 100\\%"))
check(f"shares {h['share_ge_9977']*100:.0f}/{pool['share_ge_9977']*100:.0f}/{pool['share_100']*100:.0f}%",
      has(f"{h['share_ge_9977']*100:.0f}\\% of") and has(f"{pool['share_ge_9977']*100:.0f}\\% of") and has(f"{pool['share_100']*100:.0f}\\% reach 100"))
check(f"pool mean {pct(pool['mean'])}", has(pct(pool["mean"]) + "\\%"))
lk = INF["scaling_leak"]
check("scaling leak numbers", has(f"{pct(lk['knn_proper'])}") and has(f"{pct(lk['knn_leak'])}") and has(f"{pct(lk['svm_proper'])}"))

# ---- conformal, subsets, noise
a2, a1 = CO["by_alpha"]["0.002"], CO["by_alpha"]["0.01"]
check("conformal 99.96 / 1.02 / 98.0 / 99.53", has(pct(a2["cov"])) and has(f"{a2['size']:.2f} crops")
      and has(f"{a2['single']*100:.1f}\\%") and has(f"({pct(a1['cov'])}\\% coverage)"))
ms = {tuple(k): v for k, v in CO["multi_sets_alpha_0.002"]}
check("shortlists 100/217 rice-jute, 42 lentil-mothbeans", ms[("jute", "rice")] == 100 and CO["multi_set_count_alpha_0.002"] == 217
      and ms[("lentil", "mothbeans")] == 42 and has("100 of the 217") and has("(42)"))
check("top-2 = 100%", CO["top2_accuracy"] == 1.0 and has("top two in 100\\%"))
for k, v in [("Climate (T, H, R)", "92.2"), ("Climate + pH", "96.4"), ("Climate + N, P, K", "99.3"), ("Soil (N, P, K, pH)", "79.4")]:
    check(f"subset {k} {v}", f"{SU[k]['mean']*100:.1f}" == v and has(v + "\\%"))
nm = NO["models"]
check("noise RF 96.0/87.8, ET 92.7, stacking 91.7, XGB/LGBM ~79",
      f"{nm['Random Forest'][2]*100:.1f}" == "96.0" and f"{nm['Random Forest'][3]*100:.1f}" == "87.8"
      and f"{nm['Extra Trees'][3]*100:.1f}" == "92.7" and f"{nm['Stacking'][3]*100:.1f}" == "91.7"
      and round(nm["XGBoost"][3] * 100) == 79 and round(nm["LightGBM"][3] * 100) == 79)
check("noise spread 14 points (ET vs XGBoost at 10%)", int((nm["Extra Trees"][3] - nm["XGBoost"][3]) * 100) == 14 and has("14 points"))

# ---- XAI
g = X["gini"]
check("Gini 0.222/0.215, same top-5, temperature & pH last",
      has(f"({g['rainfall']:.3f})") and has(f"({g['humidity']:.3f})") and set(X["gini_rank"][:5]) == set(X["shap_global_rank"][:5])
      and set(X["gini_rank"][-2:]) == set(X["shap_global_rank"][-2:]) == {"temperature", "ph"})
sg, sr = sum(X["shap_global"].values()), sum(X["shap_rice"].values())
check("rice SHAP 24% vs 17% global", f"{X['shap_rice']['rainfall']/sr*100:.0f}" == "24" and f"{X['shap_global']['rainfall']/sg*100:.0f}" == "17"
      and has("24\\% (17\\% globally)"))
check("LIME rainfall > 123.6, p=0.97", X["lime"][0][0].startswith("Rainfall >") and has("123.6") and has("probability 0.97"))

# ---- speed
check("ONNX 0.054 vs 13.1 ms, 241x, batch 1.3x, load 60.6 vs 8.0, stacking 30.8",
      has(f"{R['single_ms']['onnx'][0]:.3f}\\,ms") and has(f"{R['single_ms']['pkl'][0]:.1f}\\,ms") and has(f"{R['speedup']['single']:.0f}$\\times$")
      and has(f"{R['speedup']['batch_1000']:.1f}$\\times$") and has(f"{R['load_ms']['onnx'][0]:.1f}\\,ms") and has(f"{R['load_ms']['pkl'][0]:.1f}\\,ms")
      and has(f"{R['stacking_sklearn']['single_ms'][0]:.1f}\\,ms"))
check("ONNX agrees 440/440, deployed model = paper RF", R["agreement_test_rows"] == {"agree": 440, "n": 440}
      and R["deployed_check"]["deployed_vs_research_agreement"] == 440)
check("Pareto 25 trees", has("25-tree") and has(f"{PA['25']['mean']*100:.2f}") and has(f"{PS['25']['onnx_ms']:.3f}")
      and has(f"{PS['700']['onnx_ms']:.3f}") and all(v["agree_440"] for v in PS.values()))
check("sklearn latency grows with trees (0.31 / 13.5 ms)", has(f"{PS['10']['sklearn_ms']:.2f}\\,ms") and has(f"{PS['700']['sklearn_ms']:.1f}\\,ms"))

# ---- literature
for e in LIT["verified"]:
    check(f"literature {e['key']} {e['accuracy']:.2f}", f"\\cite{{{e['key']}}}" in tex and f"{e['accuracy']:.2f}" in tex)

# ---- structural rules
sec = lambda n: re.search(r"\\section\{" + n + r"\}(.*?)(\\section|\\bibliography)", tex, re.S).group(1)  # noqa: E731
check("no \\cite in Introduction (incl. contribution box)", "\\cite" not in sec("Introduction"))
check("no \\cite in Conclusion", "\\cite" not in sec("Conclusion"))
check("contribution box present", "tcolorbox" in sec("Introduction") and "Contributions" in sec("Introduction"))
kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end", tex, re.S).group(1)
check(f"<=5 keywords ({len(kw.split(','))})", len(kw.split(",")) <= 5)
check("exactly one numbered equation", tex.count("\\begin{equation}") == 1)
check("no placeholders left", not re.findall(r"%[A-Z0-9_]+%", tex))
check("no operating modes / key-gated models", not re.search(r"GPS Auto|Manual Override|operating mode|tabpfn", tex, re.I))
keys = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", tex) for k in c.split(",")}
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
check(f"all cited keys in bib (missing {sorted(keys - bibkeys)})", keys <= bibkeys)
check("every model cited", {"breiman2001", "geurts2006", "friedman2001", "chen2016", "ke2017", "prokhorenkova2018", "cortes1995",
                            "rumelhart1986", "kingma2015", "cover1967", "wolpert1992", "pedregosa2011"} <= keys)
figs = set(re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", tex))
check(f"all {len(figs)} figures exist", all((BASE / "paper" / "figs" / f).exists() for f in figs))
print(f"\n{fails} failure(s)")
sys.exit(1 if fails else 0)
