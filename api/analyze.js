export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const { image, mimeType } = req.body;
  if (!image) return res.status(400).json({ error: 'No image provided' });

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'Gemini API key not configured' });

  const prompt = `You are an expert agricultural plant pathologist with 20+ years of experience.

TASK: Carefully examine this image and determine:
1. Is this a plant/crop image at all?
2. If yes, what plant/crop is it?
3. Does it show any disease, pest damage, or is it healthy?

IMPORTANT RULES:
- If the image is NOT a plant, set isPlant to false
- If it IS a plant, give a detailed accurate disease diagnosis
- Be specific about the disease type
- Severity should reflect actual visible damage (0-100)

Respond ONLY in this exact JSON format, no markdown, no extra text:
{
  "isPlant": true,
  "disease": "<specific disease name, or Healthy Crop if no disease>",
  "crop": "<crop/plant type you identified>",
  "severity": <0-100>,
  "confidence": <0-100>,
  "description": "<2-3 specific sentences describing exactly what you see in THIS image>",
  "treatments": ["<specific treatment 1>", "<specific treatment 2>", "<specific treatment 3>"],
  "prevention": "<one specific prevention tip for this disease>",
  "notPlantMessage": "<only fill this if isPlant is false>"
}`;

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [
              {
                inline_data: {
                  mime_type: mimeType || 'image/jpeg',
                  data: image
                }
              },
              { text: prompt }
            ]
          }],
          generationConfig: {
            temperature: 0.1,
            maxOutputTokens: 1000
          }
        })
      }
    );

    const data = await response.json();

    if (data.error) {
      return res.status(500).json({ error: data.error.message });
    }

    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    const cleaned = raw.replace(/```json|```/g, '').trim();
    const result = JSON.parse(cleaned);

    return res.status(200).json(result);

  } catch (err) {
    console.error('Gemini error:', err);
    return res.status(500).json({ error: err.message });
  }
}
