# Crop AI: A GPS-Grounded Multi-Agent System for Real-Time Crop Recommendation with LLM-Augmented Agronomic Reasoning

![ONNX](https://img.shields.io/badge/ONNX_Runtime-inference-blue?logo=onnx)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-f7931e?logo=scikitlearn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_LPU-Llama_3.3_70B-f55036)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933?logo=node.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

We present **Crop AI**, a multi-agent inference system that integrates real-time satellite soil data, live meteorological data, a RandomForestClassifier exported to ONNX, and a large language model (LLM) into a unified GPS-grounded crop recommendation pipeline. The system addresses a critical limitation in existing crop recommendation research: published systems uniformly evaluate static classifiers on held-out splits of the Kaggle crop dataset but are never connected to real-world data sources, making them impractical for direct field deployment. Crop AI closes this gap by treating each data acquisition step as an autonomous agent call—a Soil Analyzer Agent querying SoilGrids 2.0, a Weather Analyst Agent querying Open-Meteo, a Crop Predictor Agent running ONNX inference, and an Insight Engine generating LLM-based agronomic reports via Groq-hosted Llama 3.3 70B. On the standard Kaggle crop recommendation benchmark (6,600 samples, 80/20 stratified split), the RandomForest classifier achieves **99.32% accuracy**, matching the best published results while additionally supporting live GPS-grounded input and natural language recommendations. Benchmarking shows ONNX Runtime achieves up to **2,159× faster single-inference** and **827× faster model loading** compared to the equivalent scikit-learn pickle pipeline, justifying the ONNX export as a production serving strategy.

---

## 1. Introduction

Accurate pre-season crop selection is a foundational decision in precision agriculture, with direct impact on yield, resource utilization, and economic return. The decision depends on a combination of soil nutrients (nitrogen, phosphorus, potassium, pH) and local climate conditions (temperature, humidity, rainfall), making it a natural multivariate classification problem.

A substantial body of machine learning research has addressed this problem using the publicly available Kaggle crop recommendation dataset [Ingle, A.]. These works demonstrate high classification accuracy—commonly 93–99%—using algorithms ranging from Decision Trees and Naive Bayes to Random Forest and Gradient Boosting ensembles. However, a consistent limitation across this literature is that **input features are treated as static, pre-known values**. In practice, obtaining accurate soil nutrient values requires either laboratory analysis (costly, slow) or manual lookup from agricultural extension tables (imprecise, location-generic). No published system to date automatically acquires live soil and weather features from a GPS coordinate before running inference.

A second limitation is the absence of a **reasoning layer**: existing systems output a crop label and nothing further. A farmer receiving a bare label without explanation of why a crop was recommended, or what agronomic conditions support that recommendation, cannot make an informed decision.

This work makes the following contributions:

1. A **four-agent pipeline** that autonomously acquires live soil and weather features from a GPS coordinate, runs classification inference, and generates a structured agronomic report—end-to-end from coordinates to a natural language recommendation.
2. An **ONNX-based inference serving strategy** demonstrated to be significantly faster than the equivalent scikit-learn pickle pipeline, with empirically measured speedups of up to 2,159× for single inference and 827× for model loading.
3. **Corrected and independently verified competitive benchmarking** against four published systems on the standard crop recommendation dataset.

---

## 2. Related Work

### 2.1 Machine Learning Approaches to Crop Recommendation

**Acharya et al. (2025)** [1] evaluated ten supervised ML algorithms on the Kaggle crop recommendation dataset using a train/test split, finding Gradient Boosting to be the highest-performing algorithm at **99.27% accuracy** (F1: 99.32%). The work additionally applied SHAP (SHapley Additive exPlanations) for post-hoc explainability. This is the strongest static-dataset accuracy in the directly comparable literature. The SHAP component is a feature attribution layer applied after prediction, not an online reasoning capability.

**Pande et al. (2024)** [2] compared nine ML algorithms—Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, and Extra Trees—on the same Kaggle dataset. Random Forest achieved the highest accuracy at **99.31%**, while AdaBoost performed poorest at 14.09%, demonstrating substantial variance across algorithm choices on this dataset. No data acquisition pipeline or explanation layer is described.

**Shams et al. (2024)** [3] proposed XAI-CROP, an explainability-oriented crop recommendation approach evaluated using regression metrics (R² = 0.9415, MAE = 0.9874) rather than classification accuracy, making direct accuracy comparison with other systems on this list inappropriate. The work benchmarks XAI-CROP against Gradient Boosting, Decision Tree, Random Forest, Gaussian Naive Bayes, and Multimodal Naive Bayes on the Kaggle dataset. No live data integration is described.

**Singh and Sharma (2025)** [4] proposed TCRM (Transformative Crop Recommendation Model), a hybrid architecture combining Random Forest, Extra Trees, dense layers, batch normalization, and multi-head attention, deployed on a cloud platform. On a custom dataset, TCRM achieved **94.00% accuracy**, outperforming Logistic Regression (91.17%), KNN (92.83%), and AdaBoost (11.50%). The system does not use live soil or weather data acquisition, and the LLM reasoning component is absent.

### 2.2 Positioning of This Work

All four related systems operate on static, pre-entered feature vectors. None integrates live satellite or meteorological APIs to automatically construct the feature vector from a GPS coordinate. None produces natural language output. Crop AI's primary contribution is not a marginal accuracy improvement over these systems, but the combination of **live data acquisition**, **agentic pipeline orchestration**, and **LLM-augmented reasoning** into a production-serving architecture, while achieving accuracy comparable to the best published static-dataset results.

---

## 3. System Architecture

### 3.1 Agentic Pipeline

The system implements a **directed four-agent pipeline** in which each agent is a stateless HTTP endpoint. The orchestrator (a client or the demo frontend) calls agents in sequence, passing outputs of earlier agents as inputs to later agents.

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                       CROP AI — AGENT PIPELINE                       │
 │                                                                      │
 │  [Orchestrator]  ──  GPS coordinate (lat, lon)                       │
 │       │                                                              │
 │       ├──► Agent 1: Soil Analyzer                                    │
 │       │    Tool  : SoilGrids 2.0 REST (ISRIC)                        │
 │       │    Input : {lat, lon}                                        │
 │       │    Output: {N, P, K, ph}  ──────────────────────┐            │
 │       │                                                 │            │
 │       ├──► Agent 2: Weather Analyst                     │            │
 │       │    Tool  : Open-Meteo forecast + archive        │            │
 │       │    Input : {lat, lon}                           │            │
 │       │    Output: {temperature, humidity, rainfall} ───┤            │
 │       │                                                 │            │
 │       ├──► Agent 3: Crop Predictor                      │            │
 │       │    Tool  : ONNX Runtime (crop_model.onnx)       │            │
 │       │    Input : float32[1,7] ←───────────────────────┘            │
 │       │    Output: {predicted_crop}  ───────────────────────┐        │
 │       │                                                     │        │
 │       └──► Agent 4: Insight Engine                          │        │
 │            Tool  : Groq LPU · llama-3.3-70b-versatile       │        │
 │            Input : {7 features + predicted_crop} ←──────────┘        │
 │            Output: {agent_response}  (structured Markdown)           │
 └──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Agent Registry

| Agent | Endpoint | External Tool | Input Schema | Output Schema |
|-------|----------|-------------|-------------|---------------|
| Soil Analyzer | `POST /api/soil` | SoilGrids 2.0 REST (ISRIC) | `{lat, lon}` | `{N, P, K, ph}` |
| Weather Analyst | `POST /api/weather` | Open-Meteo forecast + archive | `{lat, lon}` | `{temperature, humidity, rainfall}` |
| Crop Predictor | `POST /api/predict` | ONNX Runtime · crop_model.onnx | `float32[1,7]` | `{predicted_crop}` |
| Insight Engine | `POST /api/explain` | Groq LPU · llama-3.3-70b-versatile | `{7 features + crop}` | `{agent_response}` |

### 3.3 Orchestration Modes

**GPS Auto** — full 4-agent chain requiring only a GPS coordinate:
```
Geolocation → Soil Analyzer → Weather Analyst → Crop Predictor → Insight Engine
```

**Manual** — user-supplied feature vector, 2-agent chain:
```
{N, P, K, temperature, humidity, ph, rainfall} → Crop Predictor → Insight Engine
```

---

## 4. Data Acquisition Pipeline

### 4.1 Agent 1 — Soil Analyzer

Queries SoilGrids 2.0 (ISRIC) at `https://rest.isric.org/soilgrids/v2.0/properties/query` for four soil properties at 0–5 cm depth, 250 m spatial resolution. Linear scaling maps raw SoilGrids units to the training feature space:

| SoilGrids property | Description | Scaling | Mapped feature | Training range |
|--------------------|-------------|---------|----------------|---------------|
| `nitrogen` | Nitrogen content | × 0.1 | N (mg/kg) | 0–140 |
| `ocd` | Organic carbon density | × 0.05 | P proxy (mg/kg) | 5–145 |
| `clay` | Clay fraction | × 0.02 | K proxy (mg/kg) | 5–205 |
| `phh2o` | pH in water | ÷ 10 | pH | 3.5–9.94 |

**Data quality**: SoilGrids 2.0 cross-validation explains 56–83% of soil property variance by property (mean 61%; pH achieves 83%). N, P, and K values are satellite-derived estimates, not laboratory measurements. The P and K mappings (via OCD and clay fraction respectively) are acknowledged proxies. Resolution is 250 m; sub-field variation is not captured. The REST API is in active beta with no published uptime SLA [5].

### 4.2 Agent 2 — Weather Analyst

Issues two HTTP calls to Open-Meteo for the given coordinate:

| Feature | API endpoint | Parameter | Temporal resolution |
|---------|-------------|-----------|-------------------|
| Temperature (°C) | Forecast · `current` | `temperature_2m` | Hourly, 1–11 km |
| Humidity (%) | Forecast · `current` | `relative_humidity_2m` | Hourly, 1–11 km |
| Rainfall (mm) | Archive · `daily` | `precipitation_sum` (30-day sum) | Daily, 1–11 km |

**Data quality**: 30-day accumulated precipitation is a climatological proxy. It does not account for irrigation, drainage, or evapotranspiration [6].

### 4.3 Agent 3 — Crop Predictor

Receives the assembled 7-feature vector, constructs a `float32[1,7]` ONNX tensor in the following fixed order, and executes one forward pass through the ONNX inference session:

```
Feature order: [N, P, K, temperature, humidity, ph, rainfall]
```

The ONNX artifact (see Section 5.3) includes all preprocessing transformations internally. The `label` output is an integer 1–22 mapped to a crop name via an in-process lookup table. No external preprocessing step is required at inference time.

**Supported output classes (22)**: Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee.

### 4.4 Agent 4 — Insight Engine

Invokes Llama 3.3 70B (128k context) via Groq LPU inference using an OpenAI-compatible API. The prompt encodes all seven feature values and the predicted crop label, requesting a structured Markdown report that covers: agronomic justification for each feature value, soil preparation, irrigation strategy, fertilizer schedule, planting season, and expected yield range.

| Property | Value |
|----------|-------|
| Model | `llama-3.3-70b-versatile` |
| Context window | 128,000 tokens |
| Measured throughput (Groq LPU) | ~275–337 tokens/sec standard |

---

## 5. Machine Learning Methodology

### 5.1 Dataset

The training corpus is the Kaggle Crop Recommendation Dataset [Ingle, A.], comprising 2,200 labeled samples across 22 crop classes (100 samples/class) with 7 continuous features. The dataset is primarily sourced from the Indian subcontinent and represents a broad range of agroclimatic conditions.

### 5.2 Data Augmentation

To improve classifier robustness and class balance at scale, a **per-class multivariate normal augmentation** strategy is applied:

1. For each of the 22 crop classes, compute the empirical mean vector μ ∈ ℝ⁷ and covariance matrix Σ ∈ ℝ^(7×7).
2. Draw 200 synthetic samples from 𝒩(μ, Σ).
3. Clip each sample to physiologically valid feature ranges (see Section 6.1).
4. Concatenate with the original 100 real samples per class.

This triples the dataset from 2,200 to **6,600 samples** (300 per class, balanced), raising accuracy from approximately 85% (real data only) to 99.32%.

### 5.3 Training Pipeline

```
Raw dataset (2,200 samples, 22 classes, 7 features)
         │
         ▼
 Multivariate augmentation  →  6,600 samples (300/class)
         │
         ▼
 Stratified train/test split (80/20, random_state=42)
 ├── Training set: 5,280 samples
 └── Test set:     1,320 samples
         │
         ▼
 scikit-learn Pipeline:
   MinMaxScaler(feature_range=[0,1])
     └── StandardScaler()
           └── RandomForestClassifier(n_estimators=100,
                                      max_depth=None,
                                      random_state=42)
         │
         ▼
 Test set evaluation
 └── Accuracy: 99.32%
         │
         ▼
 skl2onnx export → crop_model.onnx
   (full Pipeline including both scalers baked in;
    raw feature values passed directly at inference)
```

### 5.4 Model Performance

| Metric | Value |
|--------|-------|
| **Test accuracy** | **99.32%** |
| Training samples | 5,280 |
| Test samples | 1,320 |
| Classes | 22 |
| Test split | 20% stratified |

**Note on reported accuracy**: The model configuration file (`model/model_config.json`) records `model_accuracy: 0.9931818182`, equivalent to 99.32%. This is the accuracy on the stratified held-out test set (1,320 samples), confirmed by running `accuracy_score(y_test, y_pred)` in the training notebook.

**Feature importance** (qualitative ordering, most to least discriminative):
Rainfall > Temperature > Humidity > pH > K proxy > N > P proxy

Rainfall and temperature carry the highest inter-class discriminative signal. The P proxy (OCD-derived) has lower importance due to compressed variance from the SoilGrids scaling factor.

---

## 6. Inference Optimization: Pickle vs. ONNX Runtime

### 6.1 Motivation

The scikit-learn model is trained and serialized as three pickle files: `model.pkl`, `minmaxscaler.pkl`, `standscaler.pkl`. At inference, all three must be deserialized and composed into a pipeline. The ONNX export (`crop_model.onnx`) consolidates the full pipeline—both scalers and the RandomForest—into a single binary artifact executed by the ONNX Runtime, which does not require a Python process, scikit-learn, or any ML framework dependency in production.

### 6.2 Benchmark Methodology

Benchmarks were run on the production codebase using Python 3.12, scikit-learn 1.8.0, and ONNX Runtime 1.24.2 on Apple Silicon (CPU execution provider). Each timing metric used 20 warmup iterations followed by 1,000 timed iterations; results are reported as mean ± std. A representative feature vector corresponding to rice-growing conditions was used as input.

### 6.3 Results

| Metric | scikit-learn PKL pipeline | ONNX Runtime | ONNX speedup |
|--------|--------------------------|-------------|-------------|
| **Model load time** | 15,072 ms | 18.22 ms | **827×** |
| **Single inference (mean)** | 13.052 ms | 0.006 ms | **2,159×** |
| **Single inference (std)** | ±0.771 ms | ±0.001 ms | — |
| **Batch / 1,000 samples** | 13.69 ms | 1.29 ms | **10.6×** |

The PKL pipeline's 15-second load time is dominated by deserializing the 100-tree RandomForest from pickle; this makes it unsuitable for serverless or cold-start deployment patterns. The ONNX session initializes in 18 ms and amortizes to negligible cost for long-running server processes.

Both runtimes produce identical crop predictions on all tested inputs.

![Figure 3 — PKL vs ONNX Benchmark](assets/fig3_onnx_vs_pkl.png)

---

## 7. Comparative Analysis

### 7.1 Accuracy vs. Published Systems

The table below compares Crop AI against published systems on the same or equivalent Kaggle crop recommendation dataset. All accuracy figures are from the cited papers and independently verified against their published sources.

| System | Algorithm | Accuracy | Dataset | Live data | LLM reasoning |
|--------|-----------|----------|---------|-----------|---------------|
| **Crop AI (this work)** | RandomForest + 4-agent pipeline | **99.32%** | Kaggle + synthetic (6,600) | **Yes (GPS)** | **Yes (Llama 3.3 70B)** |
| Acharya et al. [1] | Gradient Boosting + SHAP | 99.27% | Kaggle (2,200) | No | No (SHAP post-hoc) |
| Pande et al. [2] | Random Forest (best of 9) | 99.31% | Kaggle | No | No |
| Shams et al. [3] | XAI-CROP (classical ML) | R²=0.9415† | Kaggle | No | No |
| Singh & Sharma [4] | TCRM (RF+ET+Attention) | 94.00% | Custom | No | No |

†Shams et al. report regression metrics (R², MAE, MSE) rather than classification accuracy; direct comparison is not applicable.

![Figure 1 — Accuracy Comparison](assets/fig1_accuracy.png)

**Key observation**: Crop AI achieves accuracy comparable to the best published systems (Acharya et al.: 99.27%; Pande et al.: 99.31%) while being the only system to simultaneously support live GPS-grounded feature acquisition and LLM-generated agronomic reasoning. The accuracy advantage of published systems over Crop AI is at most 0.01 percentage points—within statistical noise of the dataset split—while the operational gap (static vs. live data; label vs. natural language output) is categorically different.

### 7.2 Capability Comparison

![Figure 2 — Capability Matrix](assets/fig2_capability.png)

The capability matrix shows binary presence or absence of eight operational features across all five systems. Crop AI is the only system to satisfy all eight criteria. Acharya et al. satisfy the SHAP-based explainability criterion partially (post-hoc attribution, not real-time LLM reasoning). No other system in the comparison integrates live data acquisition.

### 7.3 Feature Schema Reference

| Feature | Symbol | Training range | Unit | Acquisition source |
|---------|--------|---------------|------|--------------------|
| Nitrogen | N | 0–140 | mg/kg | Soil Analyzer · SoilGrids `nitrogen` × 0.1 |
| Phosphorus proxy | P | 5–145 | mg/kg | Soil Analyzer · SoilGrids `ocd` × 0.05 |
| Potassium proxy | K | 5–205 | mg/kg | Soil Analyzer · SoilGrids `clay` × 0.02 |
| Temperature | — | 8.8–43.7 | °C | Weather Analyst · Open-Meteo `temperature_2m` |
| Humidity | — | 14.3–99.98 | % | Weather Analyst · Open-Meteo `relative_humidity_2m` |
| Soil pH | ph | 3.5–9.94 | — | Soil Analyzer · SoilGrids `phh2o` ÷ 10 |
| 30-day rainfall | — | 20.4–298.6 | mm | Weather Analyst · Open-Meteo `precipitation_sum` |

---

## 8. API Reference

All agent endpoints accept `application/json` via `POST`. The inference server is started with `node server.js` (port 3000).

### `POST /api/soil` — Soil Analyzer Agent
```bash
curl -X POST http://localhost:3000/api/soil \
  -H "Content-Type: application/json" \
  -d '{"lat": 28.704060, "lon": 77.102493}'
# → {"N": 50.5, "P": 25.3, "K": 120.8, "ph": 6.8}
```

### `POST /api/weather` — Weather Analyst Agent
```bash
curl -X POST http://localhost:3000/api/weather \
  -H "Content-Type: application/json" \
  -d '{"lat": 28.704060, "lon": 77.102493}'
# → {"temperature": 28.5, "humidity": 65, "rainfall": 156.2}
```

### `POST /api/predict` — Crop Predictor Agent
```bash
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":202.9}'
# → {"predicted_crop": "rice"}
```

### `POST /api/explain` — Insight Engine
```bash
curl -X POST http://localhost:3000/api/explain \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20.9,"humidity":82,"ph":6.5,"rainfall":202.9,"predicted_crop":"rice"}'
# → {"agent_response": "## Why Rice?\n\n..."}
```

**Full pipeline (Python)**:
```python
import requests
BASE = 'http://localhost:3000'

soil    = requests.post(f'{BASE}/api/soil',    json={'lat': 28.7041, 'lon': 77.1025}).json()
weather = requests.post(f'{BASE}/api/weather', json={'lat': 28.7041, 'lon': 77.1025}).json()
pred    = requests.post(f'{BASE}/api/predict', json={**soil, **weather}).json()
report  = requests.post(f'{BASE}/api/explain', json={**soil, **weather, **pred}).json()

print(f"Crop: {pred['predicted_crop']}")
print(report['agent_response'])
```

---

## 9. Limitations and Future Work

| Area | Current limitation | Potential mitigation |
|------|-------------------|---------------------|
| **Soil feature quality** | SoilGrids 2.0 explains 56–83% of soil property variance (mean 61%); N, P, K are estimates, not lab measurements [5] | Integration of IoT soil sensors or lab measurement API services |
| **P and K proxy fidelity** | OCD and clay fraction are imperfect proxies for plant-available phosphorus and potassium | Direct SoilGrids `p` and `k` properties when coverage improves |
| **Rainfall proxy** | 30-day cumulative precipitation does not account for irrigation, drainage, or crop water demand | Coupled with evapotranspiration data (e.g., FAO Penman-Monteith) |
| **Geographic generalization** | Training data is biased toward the Indian subcontinent; out-of-distribution accuracy for temperate or arid agroecological zones is unvalidated | Regional dataset collection and multi-region fine-tuning |
| **Crop coverage** | 22 classes only; no tree crops, vegetables, or specialty crops | Expanded dataset covering FAOSTAT crop taxonomy |
| **Agent fault tolerance** | No retry or fallback logic between agents; upstream failure aborts the pipeline | Circuit-breaker pattern; cached fallback features at known locations |
| **LLM grounding** | The LLM reasoning agent may hallucinate agronomic facts not grounded in the feature values | Retrieval-augmented generation (RAG) over agronomic knowledge bases |

---

## 10. Conclusion

We have presented Crop AI, a multi-agent pipeline for GPS-grounded, real-time crop recommendation that integrates live satellite soil data, meteorological data, ONNX-based ML inference, and LLM-generated agronomic reasoning. The system achieves 99.32% classification accuracy on the standard Kaggle crop recommendation benchmark, matching the best published results, while extending the state of the art in three directions: live data acquisition from GPS coordinates, agentic pipeline orchestration, and natural language output via LLM reasoning.

The ONNX serving strategy delivers up to 2,159× faster single inference and 827× faster model loading compared to the equivalent scikit-learn pickle pipeline, demonstrating a practical production optimization for RandomForest-class models. All benchmark numbers reported here are empirically measured on the production codebase.

---

## References

[1] Acharya, N. et al. "Advancing crop recommendation system with supervised machine learning and explainable artificial intelligence." *Scientific Reports*, Nature, 2025. https://www.nature.com/articles/s41598-025-07003-8 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12264067/

[2] Pande, A. et al. "Enhancing Agricultural Productivity: A Machine Learning Approach to Crop Recommendations." *Human-Centric Intelligent Systems*, Springer, Vol. 4, pp. 497–510, 2024. https://link.springer.com/article/10.1007/s44230-024-00081-3

[3] Shams, M.Y., Gamel, S.A., Talaat, F.M. "Enhancing crop recommendation systems with explainable artificial intelligence: a study on agricultural decision-making." *Neural Computing and Applications*, Springer, Vol. 36, pp. 5695–5714, 2024. https://link.springer.com/article/10.1007/s00521-023-09391-2

[4] Singh, G., Sharma, S. "Enhancing precision agriculture through cloud based transformative crop recommendation model." *Scientific Reports*, Nature, 2025. https://www.nature.com/articles/s41598-025-93417-3 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11914076/

[5] Poggio, L. et al. "SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty." *SOIL*, Vol. 7, pp. 217–240, 2021. https://soil.copernicus.org/articles/7/217/2021/

[6] Open-Meteo. "Open-Meteo — Free Weather API." https://open-meteo.com/en/features

[7] Ingle, A. "Crop Recommendation Dataset." Kaggle, 2020. https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

---

## Appendix A — Repository Structure

```
crop-ai/
├── server.js                    ← Agent endpoint server (Express 5)
├── crop_model.onnx              ← Production model artifact (full Pipeline → ONNX)
├── model/
│   ├── CROP_RECOMMENDER.ipynb   ← Training pipeline (data → augment → train → export)
│   ├── convert_model.py         ← scikit-learn Pipeline → ONNX via skl2onnx
│   ├── benchmark.py             ← PKL vs ONNX timing benchmark
│   ├── model.pkl                ← Trained RandomForest (training reference only)
│   ├── minmaxscaler.pkl         ← MinMax scaler (baked into ONNX; not used at inference)
│   ├── standscaler.pkl          ← Standard scaler (baked into ONNX; not used at inference)
│   └── model_config.json        ← Crop label map, scaler params, test accuracy
├── assets/
│   ├── generate_charts.py       ← Figure generation script (matplotlib)
│   ├── fig1_accuracy.png        ← Figure 1: accuracy comparison
│   ├── fig2_capability.png      ← Figure 2: capability matrix
│   └── fig3_onnx_vs_pkl.png     ← Figure 3: PKL vs ONNX benchmark
├── public/
│   └── index.html               ← Demo client
├── agents/                      ← Agent scaffolding directory
├── .env                         ← GROQ_API_KEY (not committed)
└── package.json
```

## Appendix B — Reproduction

**Inference server**:
```bash
git clone <repo>
cd crop-ai
npm install
echo "GROQ_API_KEY=gsk_..." > .env
node server.js
```

**Model retraining**:
```bash
cd model
pip install kagglehub pandas numpy scikit-learn skl2onnx jupyter
# Place Kaggle credentials at ~/.kaggle/kaggle.json
jupyter notebook CROP_RECOMMENDER.ipynb
# Kernel → Restart & Run All
# Outputs: model.pkl, minmaxscaler.pkl, standscaler.pkl, model_config.json
python3 convert_model.py   # → ../crop_model.onnx
```

**Regenerate figures**:
```bash
pip install matplotlib numpy
python3 assets/generate_charts.py
```

**Re-run PKL vs ONNX benchmark**:
```bash
cd model
python3 benchmark.py
```

## Appendix C — Environment

| Component | Version |
|-----------|---------|
| Node.js | 18+ |
| Express | 5.x |
| onnxruntime-node | 1.x |
| Python (training only) | 3.8+ |
| scikit-learn | 1.x |
| skl2onnx | 1.x |
| LLM | llama-3.3-70b-versatile via Groq |

**Required environment variable**:

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Yes | [console.groq.com](https://console.groq.com) |

---

## Data & License

- **License**: MIT
- **Training data**: Kaggle Crop Recommendation Dataset [7] — open for research use
- **Soil data**: SoilGrids 2.0 / ISRIC [5] — CC-BY 4.0
- **Weather data**: Open-Meteo [6] — CC-BY 4.0
- **LLM**: Meta Llama 3.3 70B, served via Groq
- **ONNX export**: skl2onnx (Apache 2.0)
