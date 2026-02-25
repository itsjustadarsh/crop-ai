const axios = require('axios');

const weatherTool = {
  definition: {
    name: 'get_weather_data',
    description: 'Fetch current temperature, humidity and 30-day cumulative rainfall from GPS coordinates using Open-Meteo API',
    parameters: {
      type: 'object',
      properties: {
        lat: { type: 'number', description: 'Latitude of the location' },
        lon: { type: 'number', description: 'Longitude of the location' }
      },
      required: ['lat', 'lon']
    }
  },

  run: async ({ lat, lon }) => {
    const current = await axios.get(
      `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`
    );
    const temperature = current.data.current.temperature_2m;
    const humidity    = current.data.current.relative_humidity_2m;

    const today = new Date();
    const past  = new Date();
    past.setDate(today.getDate() - 30);
    const start = past.toISOString().split('T')[0];
    const end   = today.toISOString().split('T')[0];

    const rainRes = await axios.get(
      `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=${start}&end_date=${end}&daily=precipitation_sum`
    );
    const rainfall = rainRes.data.daily.precipitation_sum.reduce((a, b) => a + b, 0);

    return { temperature, humidity, rainfall };
  }
};

module.exports = weatherTool;
