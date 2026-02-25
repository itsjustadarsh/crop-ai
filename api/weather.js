const axios = require('axios');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { lat, lon } = req.body;

    if (!lat || !lon) {
      return res.status(400).json({ error: 'lat and lon required' });
    }

    const curr = await axios.get(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`
    );

    const today = new Date();
    const past = new Date();
    past.setDate(today.getDate() - 30);

    const rain = await axios.get(
      `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=${past.toISOString().split('T')[0]}&end_date=${today.toISOString().split('T')[0]}&daily=precipitation_sum`
    );

    res.json({
      temperature: curr.data.current.temperature_2m,
      humidity:    curr.data.current.relative_humidity_2m,
      rainfall:    rain.data.daily.precipitation_sum.reduce((a, b) => a + b, 0),
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
