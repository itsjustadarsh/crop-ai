require('dotenv').config();
const express    = require('express');
const axios      = require('axios');
const cors       = require('cors');
const ort        = require('onnxruntime-node');
const { OpenAI } = require('openai');

const app = express();
app.use(cors());
app.use(express.json());

// Initialize Groq (with explicit error handling)
let groq;
try {
  groq = new OpenAI({
    apiKey:  process.env.GROQ_API_KEY,
    baseURL: 'https://api.groq.com/openai/v1',
  });
  if (!process.env.GROQ_API_KEY) {
    console.warn('⚠️  GROQ_API_KEY not set in .env');
  } else {
    console.log('✅ Groq API initialized');
  }
} catch (e) {
  console.error('❌ Failed to initialize Groq:', e.message);
}

const CROP = {
  1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',
  8:'apple',9:'muskmelon',10:'watermelon',11:'grapes',12:'mango',13:'banana',
  14:'pomegranate',15:'lentil',16:'blackgram',17:'mungbean',18:'mothbeans',
  19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee',
};

let session;
(async () => {
  session = await ort.InferenceSession.create('./crop_model.onnx');
  console.log('✅ ONNX model loaded');
})();

app.post('/api/soil', async (req, res) => {
  try {
    const { lat, lon } = req.body;
    const { data } = await axios.get(`https://rest.isric.org/soilgrids/v2.0/properties/query?lon=${lon}&lat=${lat}&property=nitrogen&property=ocd&property=phh2o&property=clay&depth=0-5cm&value=mean`);
    const L = data.properties.layers;
    res.json({
      N:  L.find(l => l.name === 'nitrogen').depths[0].values.mean * 0.1,
      P:  L.find(l => l.name === 'ocd').depths[0].values.mean     * 0.05,
      K:  L.find(l => l.name === 'clay').depths[0].values.mean    * 0.02,
      ph: L.find(l => l.name === 'phh2o').depths[0].values.mean   / 10,
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/weather', async (req, res) => {
  try {
    const { lat, lon } = req.body;
    const curr = await axios.get(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`);
    const today = new Date(), past = new Date();
    past.setDate(today.getDate() - 30);
    const rain = await axios.get(`https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=${past.toISOString().split('T')[0]}&end_date=${today.toISOString().split('T')[0]}&daily=precipitation_sum`);
    res.json({
      temperature: curr.data.current.temperature_2m,
      humidity:    curr.data.current.relative_humidity_2m,
      rainfall:    rain.data.daily.precipitation_sum.reduce((a, b) => a + b, 0),
    });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/predict', async (req, res) => {
  try {
    const { N, P, K, temperature, humidity, ph, rainfall } = req.body;
    const tensor = new ort.Tensor('float32', new Float32Array([N, P, K, temperature, humidity, ph, rainfall]), [1, 7]);
    const result = await session.run({ float_input: tensor });
    res.json({ predicted_crop: CROP[result.label.data[0]] || 'unknown' });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.post('/api/explain', async (req, res) => {
  try {
    if (!process.env.GROQ_API_KEY) {
      return res.status(500).json({ error: 'GROQ_API_KEY not configured in .env' });
    }

    const { N, P, K, temperature, humidity, ph, rainfall, predicted_crop } = req.body;
    
    const r = await groq.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are an expert agricultural advisor. Be concise and practical.' },
        { role: 'user',   content: `Predicted crop: ${predicted_crop}\nN=${N}, P=${P}, K=${K}, Temp=${temperature}°C, Humidity=${humidity}%, pH=${ph}, Rainfall(30d)=${rainfall}mm\n\nExplain why ${predicted_crop} suits these conditions (mention each value), then give practical farming advice: soil preparation, irrigation, fertilizer, best planting season, expected yield.` },
      ],
    });
    res.json({ agent_response: r.choices[0].message.content });
  } catch (e) { 
    console.error('Groq API Error:', e.message);
    res.status(500).json({ error: e.message }); 
  }
});

// MUST be after all routes — Express 5 static middleware returns 405 for POST if placed first
app.use(express.static('public'));

app.listen(3000, () => console.log('🚀 http://localhost:3000'));
