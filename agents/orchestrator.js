const { OpenAI } = require('openai');
const soilTool       = require('./tools/soil');
const weatherTool    = require('./tools/weather');
const makePredictTool = require('./tools/predict');

const SYSTEM_PROMPT = `You are an expert agricultural AI agent. You have access to tools that fetch soil nutrient data, weather conditions, and predict the best crop for a location.

Your job:
1. Use the available tools to gather all necessary data and make a crop prediction.
2. After predicting, write a clear explanation of WHY this crop suits the specific conditions — reference each parameter (N, P, K, pH, temperature, humidity, rainfall) in your reasoning.
3. Provide practical farming advice including:
   - Soil preparation tips
   - Irrigation recommendations
   - Fertilizer guidance
   - Best planting season
   - Expected yield range

Be concise but thorough. Use plain language that a farmer can understand.`;

class CropAgent {
  constructor(session) {
    this.client = new OpenAI({
      apiKey:  process.env.GROQ_API_KEY,
      baseURL: 'https://api.groq.com/openai/v1'
    });

    const predictTool = makePredictTool(session);
    this.tools = [soilTool, weatherTool, predictTool];

    this.toolMap = {};
    for (const tool of this.tools) {
      this.toolMap[tool.definition.name] = tool.run;
    }
  }

  async run(userInput, onStep = () => {}) {
    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user',   content: userInput }
    ];

    const collectedData = {};

    while (true) {
      const response = await this.client.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        messages,
        tools: this.tools.map(t => ({ type: 'function', function: t.definition })),
        tool_choice: 'auto'
      });

      const msg = response.choices[0].message;
      messages.push(msg);

      if (!msg.tool_calls || msg.tool_calls.length === 0) {
        return { ...collectedData, agent_response: msg.content };
      }

      for (const call of msg.tool_calls) {
        onStep({ type: 'tool', tool: call.function.name });

        const args   = JSON.parse(call.function.arguments);
        const result = await this.toolMap[call.function.name](args);

        if (typeof result === 'object' && result !== null) {
          Object.assign(collectedData, result);
        } else if (call.function.name === 'predict_crop') {
          collectedData.predicted_crop = result;
        }

        // After predict_crop the next LLM call generates the final analysis
        if (call.function.name === 'predict_crop') {
          onStep({ type: 'analyzing' });
        }

        messages.push({
          role:         'tool',
          tool_call_id: call.id,
          content:      JSON.stringify(result)
        });
      }
    }
  }
}

module.exports = CropAgent;
