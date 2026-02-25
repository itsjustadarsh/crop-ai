const ort = require('onnxruntime-node');

const crop_dict = {
  1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',
  8:'apple',9:'muskmelon',10:'watermelon',11:'grapes',12:'mango',13:'banana',
  14:'pomegranate',15:'lentil',16:'blackgram',17:'mungbean',18:'mothbeans',
  19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee'
};

function makePredictTool(session) {
  return {
    definition: {
      name: 'predict_crop',
      description: 'Predict the best crop to grow given soil nutrients and weather conditions',
      parameters: {
        type: 'object',
        properties: {
          N:           { type: 'number', description: 'Nitrogen content in soil (kg/ha)' },
          P:           { type: 'number', description: 'Phosphorus content in soil (kg/ha)' },
          K:           { type: 'number', description: 'Potassium content in soil (kg/ha)' },
          temperature: { type: 'number', description: 'Temperature in Celsius' },
          humidity:    { type: 'number', description: 'Relative humidity in percent' },
          ph:          { type: 'number', description: 'Soil pH value' },
          rainfall:    { type: 'number', description: '30-day cumulative rainfall in mm' }
        },
        required: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
      }
    },

    run: async ({ N, P, K, temperature, humidity, ph, rainfall }) => {
      const input   = new Float32Array([N, P, K, temperature, humidity, ph, rainfall]);
      const tensor  = new ort.Tensor('float32', input, [1, 7]);
      const results = await session.run({ float_input: tensor });
      const label   = results.label.data[0];
      return crop_dict[label] || 'unknown';
    }
  };
}

module.exports = makePredictTool;
