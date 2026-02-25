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

    const { data } = await axios.get(
      `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=${lon}&lat=${lat}&property=nitrogen&property=ocd&property=phh2o&property=clay&depth=0-5cm&value=mean`
    );

    const L = data.properties.layers;
    res.json({
      N:  L.find(l => l.name === 'nitrogen').depths[0].values.mean * 0.1,
      P:  L.find(l => l.name === 'ocd').depths[0].values.mean     * 0.05,
      K:  L.find(l => l.name === 'clay').depths[0].values.mean    * 0.02,
      ph: L.find(l => l.name === 'phh2o').depths[0].values.mean   / 10,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
};
