# Reproducing every result in the paper

Everything uses open-source libraries from PyPI and a public Kaggle dataset. No accounts, API keys or licence-gated model weights are required. Every random step is seeded with 42.

## 1. Environment
```bash
python3.14 -m venv benchmark/.venv
benchmark/.venv/bin/pip install -r benchmark/requirements.txt
```

## 2. Dataset (public, no login needed)
```bash
benchmark/.venv/bin/python -c "import kagglehub; print(kagglehub.dataset_download('atharvaingle/crop-recommendation-dataset'))"
shasum -a 256 ~/.cache/kagglehub/datasets/atharvaingle/crop-recommendation-dataset/versions/1/Crop_recommendation.csv
# expected: 54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e  (2,200 rows + header)
```

## 3. Experiments (in order)
| Step | Command | Produces |
|---|---|---|
| Baseline 6 models, raw and augmented; RF winner | `python benchmark/onnx_winner_benchmark.py` | `paper/assets/onnx_winner_results.json`, `model/research_best_model.pkl` |
| CV-tuned ensembles | `python benchmark/tuned_ensembles.py` | `paper/assets/tuned_ensembles.json` |
| Accuracy push, locked split | `PUSH_OUT=accuracy_push_locked.json python benchmark/accuracy_push.py locked` | `paper/assets/accuracy_push_locked.json` |
| Accuracy push, 5×10 repeated CV and ceiling | `PUSH_OUT=accuracy_push_repeated.json python benchmark/accuracy_push.py repeated` then `... ceiling` | `paper/assets/accuracy_push_repeated.json` |
| Ceiling (per-sample errors across models) | `python benchmark/ceiling_analysis.py` | `paper/assets/ceiling.json` |
| Data audit, inflation, conformal shortlists, input subsets, noise, significance, trees-vs-accuracy | `python benchmark/novelty_experiments.py synthetic inflation conformal subsets noise significance pareto_acc` | `paper/assets/novelty_*.json` |
| Trees-vs-latency (idle machine) | `python benchmark/novelty_experiments.py pareto_speed` | `paper/assets/novelty_pareto_speed.json` |
| Fill the paper's computed numbers | `python benchmark/fill_paper.py` | updates `paper/paper.tex` |
| ONNX speed (run on an otherwise idle machine) | `python benchmark/onnx_runtime_benchmark.py` | `paper/assets/onnx_runtime_results.json` |
| Figures and XAI values | `python benchmark/paper_figures.py all` | `paper/figs/*.pdf`, `paper/assets/xai_results.json` |
| Check the paper against all outputs | `python benchmark/audit_paper.py` | PASS/FAIL list |
| Build the PDF | `cd paper && tectonic -X compile paper.tex --outdir build` | `paper/build/paper.pdf` |

Run every `python` above with `benchmark/.venv/bin/python` from the repository root.

## Expected variation
- **Accuracy numbers:** should reproduce exactly with the pinned versions. Different library versions or CPU types can change a tree split, and with it a single prediction.
- **Timings:** hardware-specific (reported on an Apple M5, 1 thread). Absolute milliseconds will differ; the speed-up ratios should be similar.
