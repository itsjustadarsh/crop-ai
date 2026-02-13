# Crop AI

A complete machine learning solution for recommending crops based on soil nutrients and weather parameters. Features both **automated GPS-based analysis** and **manual input modes**. Trained on real Kaggle data with synthetic augmentation, achieving **98.56% accuracy**.

## Quick Start

```bash
# Install dependencies
npm install

# Start the Crop AI server
node server.js
# Server runs at http://localhost:3000
```

## Project Overview

This project provides:
- � **GPS-Based Mode**: Auto-detect soil & weather from coordinates
- ✍️ **Manual Mode**: Direct input for custom parameters
- 🎯 **Real Kaggle Dataset**: 2,200 verified crop samples
- 🔄 **Data Augmentation**: 4,400 synthetic samples (3x augmentation)
- 🤖 **ML Model**: RandomForestClassifier with 98.56% accuracy
- 📦 **ONNX Format**: Optimized cross-platform model inference
- 🌾 **Crop Support**: 22 different crop varieties
- 🚀 **Production Ready**: Express.js REST API

## Project Structure

```
crop-ai/
├── model/
│   ├── CROP_RECOMMENDER.ipynb        ← Training/generation notebook
│   ├── convert_model.py              ← Convert pickle to ONNX format
│   ├── model.pkl                     ← Trained RandomForest classifier (8.8 MB)
│   ├── minmaxscaler.pkl              ← MinMax normalization scaler
│   ├── standscaler.pkl               ← Standard scaling scaler
│   ├── model_config.json             ← Model configuration & metadata
│   └── backup/                       ← Previous model versions (gitignored)
├── server.js                         ← Express.js REST API server
├── public/
│   └── index.html                    ← Web interface
├── package.json                      ← Node.js dependencies
├── crop_model.onnx                   ← Optimized ONNX model for inference
└── README.md                         ← This file
```

## How It Works

### System Architecture

```
Crop AI Server (Node.js + Express)
├── ONNX Model (crop_model.onnx)
├── GPS Mode (/predict-auto)
│   ├── SoilGrids API → Soil data (N, P, K, pH)
│   └── Open-Meteo API → Weather (temperature, humidity, rainfall)
└── Manual Mode (/predict-manual)
    └── Direct input parameters
        ↓
    Prediction → CropID → Crop Name
```

### Prediction Pipeline

1. **Get Input Data**
   - **Auto Mode**: Fetch from GPS coordinates via APIs
   - **Manual Mode**: Direct JSON input

2. **Process Features**
   - Extract: N, P, K, Temperature, Humidity, pH, Rainfall
   - Normalize to feature ranges

3. **Inference**
   - Load ONNX model
   - Run prediction in ONNX Runtime
   - Output: Crop ID (1-22)

4. **Return Result**
   - Map ID to crop name
   - Return with input parameters and prediction

### Input Features (7 total)
| Feature | Range | Unit |
|---------|-------|------|
| **N** (Nitrogen) | 0-140 | mg/kg |
| **P** (Phosphorus) | 5-145 | mg/kg |
| **K** (Potassium) | 5-205 | mg/kg |
| **Temperature** | 8.8-43.7 | °C |
| **Humidity** | 14.3-99.98 | % |
| **pH** | 3.5-9.94 | scale |
| **Rainfall** | 20.4-298.6 | mm/year |

### Output Labels (22 crops)
Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

## Training & Model Regeneration

The `CROP_RECOMMENDER.ipynb` notebook handles the complete training pipeline for those who want to retrain the model:

1. **Load Real Data** - Downloads 2,200 samples from Kaggle
2. **Generate Synthetic Data** - Creates 4,400 augmented samples
3. **Prepare Features** - Encodes labels and organizes data
4. **Train Models** - Fits MinMaxScaler, StandardScaler, RandomForest
5. **Convert to ONNX** - Uses `convert_model.py` to export optimized format
6. **Save Models** - Exports pickle files with backups

### Running the Notebook

```bash
cd model
jupyter notebook CROP_RECOMMENDER.ipynb
# Run all cells or use Kernel → Restart & Run All
```

> **Note**: The server uses pre-trained `crop_model.onnx` for inference. Retraining only needed if dataset updates.

## REST API Endpoints

### 1. Auto Mode - GPS-Based Prediction
**`POST /predict-auto`** - Automatically fetch soil and weather from GPS coordinates

```json
{
  "lat": 28.704060,
  "lon": 77.102493
}
```

**Response**:
```json
{
  "N": 50.5,
  "P": 25.3,
  "K": 120.8,
  "temperature": 28.5,
  "humidity": 65,
  "ph": 6.8,
  "rainfall": 156.2,
  "predicted_crop": "rice"
}
```

**Data Sources**:
- **SoilGrids API** (ISRIC): Soil nitrogen, phosphorus, potassium, pH at 0-5cm depth
- **Open-Meteo API**: Current temperature, humidity, and 30-day precipitation average

### 2. Manual Mode - Direct Input
**`POST /predict-manual`** - Submit custom parameters directly

```json
{
  "N": 50,
  "P": 50,
  "K": 50,
  "temperature": 25,
  "humidity": 70,
  "ph": 6.5,
  "rainfall": 150
}
```

**Response**:
```json
{
  "N": 50,
  "P": 50,
  "K": 50,
  "temperature": 25,
  "humidity": 70,
  "ph": 6.5,
  "rainfall": 150,
  "predicted_crop": "rice"
}
```

## Using the API

### Start the Server

```bash
npm install
node server.js
# Server listens at http://localhost:3000
```

### Example: cURL

**Auto Mode** (GPS):
```bash
curl -X POST http://localhost:3000/predict-auto \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 28.704060,
    "lon": 77.102493
  }'
```

**Manual Mode**:
```bash
curl -X POST http://localhost:3000/predict-manual \
  -H "Content-Type: application/json" \
  -d '{
    "N": 50, "P": 50, "K": 50,
    "temperature": 25, "humidity": 70,
    "ph": 6.5, "rainfall": 150
  }'
```

### Example: JavaScript/Node

```javascript
// Auto mode with GPS
fetch('http://localhost:3000/predict-auto', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    lat: 28.704060,
    lon: 77.102493
  })
})
.then(r => r.json())
.then(data => console.log(`Crop: ${data.predicted_crop}`));

// Manual mode
fetch('http://localhost:3000/predict-manual', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    N: 50, P: 50, K: 50,
    temperature: 25, humidity: 70,
    ph: 6.5, rainfall: 150
  })
})
.then(r => r.json())
.then(data => console.log(`Crop: ${data.predicted_crop}`));
```

### Example: Python

```python
import requests

# Auto mode
response = requests.post('http://localhost:3000/predict-auto', json={
    'lat': 28.704060,
    'lon': 77.102493
})
print(response.json()['predicted_crop'])

# Manual mode
response = requests.post('http://localhost:3000/predict-manual', json={
    'N': 50, 'P': 50, 'K': 50,
    'temperature': 25, 'humidity': 70,
    'ph': 6.5, 'rainfall': 150
})
print(response.json()['predicted_crop'])
```

## Model Files

### Production Inference

**crop_model.onnx** (ONNX Format)
- **Purpose**: Production-grade model inference
- **Size**: Optimized for fast execution
- **Runtime**: ONNX Runtime (Node.js via `onnxruntime-node`)
- **Used By**: `server.js` for all predictions
- **Advantage**: Cross-platform compatibility, hardware acceleration support

### Training & Reference

**CROP_RECOMMENDER.ipynb** (Jupyter Notebook)
- Training pipeline with full documentation
- Data loading, augmentation, preprocessing, model building
- Saves outputs as pickle files and generates ONNX model

**convert_model.py** (Python Script)
- Converts trained pickle models to ONNX format
- Optimizes model for inference performance
- Run automatically by notebook or manually

**model_config.json** (Configuration)
- Model metadata and training statistics
- Feature names and ranges
- Crop ID to name mapping (1-22)
- Training info: 6,600 total samples (2,200 real + 4,400 synthetic)

### Training Models (Pickle Format)

**model.pkl** (8.8 MB)
- Trained RandomForestClassifier (100 trees)
- Training reference only (production uses ONNX)
- 22 crop classes, 7 input features

**minmaxscaler.pkl** (0.7 KB)
- Feature normalization [0, 1]
- Used during training; ONNX model includes preprocessing

**standscaler.pkl** (0.6 KB)
- Z-score standardization
- Used during training; ONNX model includes preprocessing

## Data Augmentation Strategy

The training process creates synthetic data to improve model robustness:

1. **Real Data**: 2,200 samples (100 per crop)
2. **Augmentation**: 
   - Group by crop class
   - Calculate mean & covariance per crop
   - Sample from multivariate normal distribution
   - Clip to realistic ranges
3. **Result**: 4,400 synthetic samples (200 per crop)
4. **Combined**: 6,600 total samples (300 per crop)

This ensures balanced classes and improves model performance from ~85% to **98.56% accuracy**.

## Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.56% |
| **Training Samples** | 5,280 |
| **Test Samples** | 1,320 |
| **Precision (macro avg)** | 0.986 |
| **Recall (macro avg)** | 0.986 |
| **F1-Score (macro avg)** | 0.986 |

### Per-Class Performance
- Most crops: 98-100% accuracy
- Few challenging crops: 93-98% accuracy (rice, jute, etc.)
- All crops above 90% threshold

## Data Source

**Dataset**: Crop Recommendation Dataset  
**Source**: [Kaggle - atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
**Samples**: 2,200  
**Features**: 7 environmental parameters  
**License**: Open for research & educational use

## Requirements

### To Run the Server
```bash
npm install
# Dependencies: express, onnxruntime-node, cors, axios
```

**Requirements**:
- Node.js 14+ and npm
- ~50 MB disk space (ONNX model + dependencies)

### To Train/Retrain the Model
```bash
cd model
pip install kagglehub pandas numpy scikit-learn jupyter
```

**Requirements**:
- Python 3.8+
- Virtual environment (venv)
- Internet access for Kaggle dataset download
- ~500 MB disk space (training data + models)

## Configuration

### Server Port

Modify in `server.js`:
```javascript
app.listen(3000, ()=> ...);  // Change 3000 to your desired port
```

### Kaggle Authentication (For Model Retraining)

Required only if retraining the model:
1. Create account at kaggle.com
2. Download API credentials from account settings
3. Place in `~/.kaggle/kaggle.json`
4. Run notebook - automatic download of 2,200 samples

**Fallback**: Offline mode creates realistic synthetic data

### Model Parameters (For Retraining)

Edit in `CROP_RECOMMENDER.ipynb`:
```python
# Data augmentation factor (currently 3x)
augmentation_factor = 3

# RandomForest parameters
n_estimators = 100
max_depth = None
random_state = 42
```



## Troubleshooting

### Issue: "Module not found" errors
```bash
# Install missing packages
pip install kagglehub pandas numpy scikit-learn
```

### Issue: Kaggle authentication fails
```bash
# Set credentials manually
export KAGGLE_USERNAME=your_username
export KAGGLE_PASSWORD=your_password
```

### Issue: Low accuracy on new data
- Ensure input ranges match training data (see Input Features section)
- Verify feature order: [N, P, K, temp, humidity, ph, rainfall]
- Check for data preprocessing errors

### Issue: API server won't start
```bash
# Check port availability
lsof -i :3000

# Try different port
PORT=3001 node server.js
```

## Performance Optimization

### Batch Predictions
```python
# Process multiple samples efficiently
samples = np.array([[50,50,50,25,70,6.5,150], [60,40,40,30,60,7,200]])
scaled = standard.transform(minmax.transform(samples))
predictions = model.predict(scaled)
```

### Using ONNX Model
```bash
# For edge deployment, use the ONNX format
# crop_model.onnx provides cross-platform compatibility
```

## Model Improvement Ideas

1. **Hyperparameter Tuning**: Grid search for optimal RandomForest parameters
2. **Feature Engineering**: Add derived features (NPK ratios, temperature variance)
3. **Ensemble Methods**: Combine with other classifiers (XGBoost, SVM)
4. **Real-time Retraining**: Incorporate feedback from actual recommendations
5. **Regional Adaptation**: Create region-specific models

## License & Attribution

- **Model**: Educational and research use
- **Dataset**: Kaggle open license
- **Code**: Available for modification and redistribution

## Support & Contact

For issues or questions:
1. Check troubleshooting section above
2. Review model configuration files
3. Run validation cells in notebook
4. Verify input data format and ranges

---

**Crop AI** - Smart crop recommendation using ML & real-time data  
**Status**: Production Ready ✅  
**Model Accuracy**: 98.56%  
**Deployed With**: Express.js + ONNX Runtime  
**Training Data**: 6,600 samples (2,200 real + 4,400 synthetic)  
**Supported Crops**: 22 varieties  
**Last Updated**: February 13, 2026
