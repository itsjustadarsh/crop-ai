const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const CROP = {
  1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',
  8:'apple',9:'muskmelon',10:'watermelon',11:'grapes',12:'mango',13:'banana',
  14:'pomegranate',15:'lentil',16:'blackgram',17:'mungbean',18:'mothbeans',
  19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee',
};

let session = null;

async function loadModel() {
  if (session) return session;
  
  try {
    const modelPath = path.join(process.cwd(), 'crop_model.onnx');
    session = await ort.InferenceSession.create(modelPath);
    console.log('✅ ONNX model loaded');
    return session;
  } catch (e) {
    console.error('Model loading failed:', e);
    throw new Error('ONNX model failed to load: ' + e.message);
  }
}

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { N, P, K, temperature, humidity, ph, rainfall } = req.body;

    if (N === undefined || P === undefined || K === undefined || 
        temperature === undefined || humidity === undefined || 
        ph === undefined || rainfall === undefined) {
      return res.status(400).json({ error: 'Missing required features: N, P, K, temperature, humidity, ph, rainfall' });
    }

    const sess = await loadModel();
    const tensor = new ort.Tensor('float32', new Float32Array([N, P, K, temperature, humidity, ph, rainfall]), [1, 7]);
    const result = await sess.run({ float_input: tensor });
    
    res.json({ predicted_crop: CROP[result.label.data[0]] || 'unknown' });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
