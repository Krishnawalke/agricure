export default async function handler(req, res) {
  // Allow CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { image, mimeType } = req.body;
  if (!image) return res.status(400).json({ error: 'No image provided' });

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'API key not configured' });

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: mimeType || 'image/jpeg',
                data: image
              }
            },
            {
              type: 'text',
              text: `You are an expert agricultural plant pathologist with 20+ years of experience.

TASK: Carefully examine this image and determine:
1. Is this a plant/crop image at all?
2. If yes, what plant/crop is it?
3. Does it show any disease, pest damage, or is it healthy?

IMPORTANT RULES:
- If the image is NOT a plant (e.g. a person, animal, object, screenshot, code, etc.), respond with isPlant: false
- If it IS a plant, give a detailed, accurate disease diagnosis
- Be specific - different diseases look very different (rust vs blight vs mosaic virus vs nutrient deficiency)
- Severity should reflect actual visible damage (0-100)

Respond ONLY in this exact JSON format, no markdown, no extra text:
{
  "isPlant": true or false,
  "disease": "<specific disease name, or 'Healthy Crop' if no disease>",
  "crop": "<crop/plant type you identified>",
  "severity": <0-100, where 0=healthy, 100=severe>,
  "confidence": <0-100, how confident you are>,
  "description": "<2-3 specific sentences describing exactly what you see in THIS image>",
  "treatments": ["<specific treatment 1>", "<specific treatment 2>", "<specific treatment 3>"],
  "prevention": "<one specific prevention tip for this disease>",
  "notPlantMessage": "<if not a plant, friendly message explaining this>"
}`
            }
          ]
        }]
      })
    });

    const data = await response.json();

    if (data.error) {
      return res.status(500).json({ error: data.error.message });
    }

    const raw = data.content.map(i => i.text || '').join('');
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const result = JSON.parse(cleaned);

    return res.status(200).json(result);

  } catch (err) {
    console.error('Analyze error:', err);
    return res.status(500).json({ error: err.message });
  }
}
