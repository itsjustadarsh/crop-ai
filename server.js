const express = require("express");
const axios = require("axios");
const cors = require("cors");
const ort = require("onnxruntime-node");

const app = express();
app.use(cors());
app.use(express.json());

let session;
(async () => {
  session = await ort.InferenceSession.create("./crop_model.onnx");
  console.log("✅ ONNX Model Loaded");
})();

const crop_dict = {
  1:'rice',2:'maize',3:'jute',4:'cotton',5:'coconut',6:'papaya',7:'orange',
  8:'apple',9:'muskmelon',10:'watermelon',11:'grapes',12:'mango',13:'banana',
  14:'pomegranate',15:'lentil',16:'blackgram',17:'mungbean',18:'mothbeans',
  19:'pigeonpeas',20:'kidneybeans',21:'chickpea',22:'coffee'
};

// 🔮 function to run ONNX prediction
async function predictCropFromFeatures(features){
  const input = new Float32Array(features);
  const tensor = new ort.Tensor("float32", input, [1,7]);
  const results = await session.run({ float_input: tensor });
  const prediction = results.label.data[0];
  return crop_dict[prediction];
}

// 🌍 AUTO MODE (GPS)
app.post("/predict-auto", async (req,res)=>{
  try{
    const {lat,lon} = req.body;
    console.log({lat,lon});
    // Soil
    const soilURL=`https://rest.isric.org/soilgrids/v2.0/properties/query?lon=${lon}&lat=${lat}&property=nitrogen&property=ocd&property=phh2o&property=clay&depth=0-5cm&value=mean`;
    const soilRes=await axios.get(soilURL);
    const layers=soilRes.data.properties.layers;

    const N=layers.find(l=>l.name==="nitrogen").depths[0].values.mean*0.1;
    const P=layers.find(l=>l.name==="ocd").depths[0].values.mean*0.05;
    const K=layers.find(l=>l.name==="clay").depths[0].values.mean*0.02;
    const ph=layers.find(l=>l.name==="phh2o").depths[0].values.mean/10;

    // Weather
    const weather=await axios.get(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m`);
    const temperature=weather.data.current.temperature_2m;
    const humidity=weather.data.current.relative_humidity_2m;

    const today=new Date();
    const past=new Date(); past.setDate(today.getDate()-30);
    const start=past.toISOString().split("T")[0];
    const end=today.toISOString().split("T")[0];

    const rainRes=await axios.get(`https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=${start}&end_date=${end}&daily=precipitation_sum`);
    const rainfall=rainRes.data.daily.precipitation_sum.reduce((a,b)=>a+b,0);

    const crop=await predictCropFromFeatures([N,P,K,temperature,humidity,ph,rainfall]);
    console.log({N,P,K,temperature,humidity,ph,rainfall,predicted_crop:crop})
    res.json({N,P,K,temperature,humidity,ph,rainfall,predicted_crop:crop});
  }catch(err){res.status(500).json({error:err.message})}
});

// ✍️ MANUAL MODE
app.post("/predict-manual", async (req,res)=>{
  try{
    const {N,P,K,temperature,humidity,ph,rainfall}=req.body;
    const crop=await predictCropFromFeatures([N,P,K,temperature,humidity,ph,rainfall]);
    res.json({N,P,K,temperature,humidity,ph,rainfall,predicted_crop:crop});
  }catch(err){res.status(500).json({error:err.message})}
});

app.listen(3000,()=>console.log("🚀 Server running at port 3000"));
