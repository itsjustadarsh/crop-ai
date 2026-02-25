const axios = require('axios');

const soilTool = {
  definition: {
    name: 'get_soil_data',
    description: 'Fetch soil nutrient data (N, P, K, pH) from GPS coordinates using the SoilGrids API',
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
    const url = `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=${lon}&lat=${lat}&property=nitrogen&property=ocd&property=phh2o&property=clay&depth=0-5cm&value=mean`;
    const res = await axios.get(url);
    const layers = res.data.properties.layers;

    const N  = layers.find(l => l.name === 'nitrogen').depths[0].values.mean * 0.1;
    const P  = layers.find(l => l.name === 'ocd').depths[0].values.mean * 0.05;
    const K  = layers.find(l => l.name === 'clay').depths[0].values.mean * 0.02;
    const ph = layers.find(l => l.name === 'phh2o').depths[0].values.mean / 10;

    return { N, P, K, ph };
  }
};

module.exports = soilTool;
