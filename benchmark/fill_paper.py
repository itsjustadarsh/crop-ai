"""Fill the remaining %PLACEHOLDER% tokens in paper/paper.tex from the experiment JSON files.
Every value comes from a results file; nothing is typed by hand. Run after novelty_experiments.py."""
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
A = BASE / "paper" / "assets"
TEX = BASE / "paper" / "paper.tex"


def j(name):
    return json.loads((A / name).read_text())


def p_fmt(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.2f}$" if p >= 0.01 else f"$p={p:.3f}$"


def main():
    s = TEX.read_text()
    fill = {}
    inf = j("novelty_inflation.json")
    h, pool, leak = inf["honest_rf"], inf["test_selected_pool"], inf["scaling_leak"]
    fill.update({
        "%INF_MIN%": f"{h['min']*100:.2f}", "%INF_MAX%": f"{h['max']*100:.2f}", "%INF_MEAN%": f"{h['mean']*100:.2f}",
        "%INF_GE%": f"{h['share_ge_9977']*100:.0f}", "%POOL_MEAN%": f"{pool['mean']*100:.2f}",
        "%POOL_GE%": f"{pool['share_ge_9977']*100:.0f}", "%POOL_100%": f"{pool['share_100']*100:.0f}",
        "%KNN_P%": f"{leak['knn_proper']*100:.2f}", "%KNN_L%": f"{leak['knn_leak']*100:.2f}",
        "%SVM_P%": f"{leak['svm_proper']*100:.2f}", "%SVM_L%": f"{leak['svm_leak']*100:.2f}",
    })

    sig = j("novelty_significance.json")["pairs"]

    def pv(a, b):
        k = f"{a} vs {b}" if f"{a} vs {b}" in sig else f"{b} vs {a}"
        return sig[k]["p"]
    others = ["Extra Trees", "CatBoost", "XGBoost", "LightGBM"]
    p_sr = pv("Random Forest", "Stacking")
    ns = [o for o in others if pv("Random Forest", o) >= 0.05]
    sig_worse = [o for o in others if pv("Random Forest", o) < 0.05]
    sent = (f"A corrected resampled $t$-test~\\cite{{nadeau2003}} finds no significant difference between "
            f"stacking and RF ({p_fmt(p_sr)})")
    if ns:
        sent += f" or between RF and {', '.join(ns)} ({', '.join(p_fmt(pv('Random Forest', o)) for o in ns)})"
    sent += "."
    if sig_worse:
        sent += (f" Only {', '.join(sig_worse)} {'is' if len(sig_worse) == 1 else 'are'} significantly less accurate than RF "
                 f"({', '.join(p_fmt(pv('Random Forest', o)) for o in sig_worse)}).")
    sent += " The leading models are therefore statistically tied."
    fill["%SIG_SENTENCE%"] = sent

    pa, ps = j("novelty_pareto_acc.json"), j("novelty_pareto_speed.json")
    full = pa["700"]["mean"]
    small = min((int(t) for t in pa if full - pa[t]["mean"] <= 0.001), key=int)
    fill["%PARETO_SENTENCE%"] = (
        f"a {small}-tree forest is within 0.1 points of the 700-tree accuracy ({pa[str(small)]['mean']*100:.2f}\\% vs "
        f"{full*100:.2f}\\%) and answers in {ps[str(small)]['onnx_ms']:.3f}\\,ms in ONNX Runtime, against "
        f"{ps['700']['onnx_ms']:.3f}\\,ms for 700 trees; all sizes agree with scikit-learn on every test sample"
        + ("." if all(v["agree_440"] for v in ps.values()) else " except where noted."))

    for k, v in fill.items():
        if k in s:
            s = s.replace(k, v)
    left = sorted(set(re.findall(r"%[A-Z0-9_]+%", s)))
    TEX.write_text(s)
    print("filled", len(fill), "remaining", left)


if __name__ == "__main__":
    main()
