# Crop AI

Code and results for the paper **"Beyond the 99% Ceiling: Rethinking the Evaluation and Deployment of Crop Recommendation Systems"**, together with the GPS-to-advice web app it describes.

## What's here
| Path | Contents |
|---|---|
| `benchmark/` | All experiments behind the paper: leakage-free model selection, tuned ensembles, repeated cross-validation, data audit, split-luck study, conformal crop shortlists, input subsets, noise robustness, significance tests and ONNX speed benchmarks |
| `benchmark/REPRODUCE.md` | Step-by-step commands to reproduce every number and figure |
| `benchmark/requirements.txt` | Exact library versions (open source only; no API keys or gated models) |
| `paper/assets/*.json` | The result files every number in the paper comes from |
| `server.js`, `api/`, `public/` | The web app: GPS point → SoilGrids soil data → Open-Meteo weather → ONNX Random Forest → plain-language advice |
| `crop_model.onnx` | The deployed model: the paper's Random Forest with its scalers, as one ONNX graph |
| `benchmark/legacy/` | Earlier experiment scripts kept for history; not used by the paper |

## Key results
- The dataset is consistent with independent, uniform sampling per crop, which explains why published accuracies cluster near 100%.
- With the test split isolated from every choice, a stacking ensemble reaches 99.77% (439/440), and Random Forest leads over 50 repeated folds at 99.53%. The top six models are statistically tied.
- The identical Random Forest scores between 98.86% and 100% depending only on the random split.
- Conformal shortlists cover 99.96% of cases with 1.02 crops on average.
- Weather data alone gives 92.2%.
- ONNX Runtime serves the Random Forest 241× faster than scikit-learn for single predictions.

## Reproduce the paper
See [`benchmark/REPRODUCE.md`](benchmark/REPRODUCE.md). In short:
```bash
python3.14 -m venv benchmark/.venv
benchmark/.venv/bin/pip install -r benchmark/requirements.txt
benchmark/.venv/bin/python benchmark/onnx_winner_benchmark.py   # then the other steps in REPRODUCE.md
```

## Run the app
```bash
npm install
echo "GROQ_API_KEY=your_key" > .env    # only needed for the advice step
npm run dev                            # http://localhost:3000
```
The soil, weather and prediction steps need no keys. The advice step calls Llama 3.3 70B through Groq. The P and K soil inputs are approximations from SoilGrids and have not been validated against laboratory tests (see the paper's limitations).

## Data
[Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) by Atharva Ingle (Kaggle, public). `REPRODUCE.md` gives its SHA-256 checksum.
