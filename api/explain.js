const { OpenAI } = require('openai');

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    if (!process.env.GROQ_API_KEY) {
      return res.status(500).json({ error: 'GROQ_API_KEY environment variable not configured' });
    }

    const groq = new OpenAI({
      apiKey:  process.env.GROQ_API_KEY,
      baseURL: 'https://api.groq.com/openai/v1',
    });

    const { N, P, K, temperature, humidity, ph, rainfall, predicted_crop } = req.body;

    if (!predicted_crop) {
      return res.status(400).json({ error: 'predicted_crop required' });
    }

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
};
