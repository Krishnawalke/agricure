export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { image, mimeType } = req.body;
    if (!image) return res.status(400).json({ error: 'No image provided' });

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return res.status(500).json({ error: 'Gemini API key not configured' });

    const prompt = `You are an expert agricultural plant pathologist.
Analyze this image and respond ONLY in this exact JSON format, no markdown, no extra text:
{"isPlant":true,"disease":"<disease name or Healthy Crop>","crop":"<crop type>","severity":<0-100>,"confidence":<0-100>,"description":"<2 sentences>","treatments":["<t1>","<t2>","<t3>"],"prevention":"<tip>","notPlantMessage":""}`;

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-002:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [
              { inline_data: { mime_type: mimeType || 'image/jpeg', data: image } },
              { text: prompt }
            ]
          }],
          generationConfig: { temperature: 0.1, maxOutputTokens: 500 }
        })
      }
    );

    // Get raw text first to debug
    const rawText = await response.text();
    
    let data;
    try {
      data = JSON.parse(rawText);
    } catch(e) {
      console.error('Gemini raw response:', rawText.substring(0, 200));
      return res.status(500).json({ error: 'Gemini API error: ' + rawText.substring(0, 100) });
    }

    if (data.error) {
      return res.status(500).json({ error: data.error.message });
    }

    const raw = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    const cleaned = raw.replace(/```json|```/g, '').trim();
    
    let result;
    try {
      result = JSON.parse(cleaned);
    } catch(e) {
      return res.status(500).json({ error: 'Could not parse AI response: ' + cleaned.substring(0, 100) });
    }

    return res.status(200).json(result);

  } catch (err) {
    console.error('Handler error:', err);
    return res.status(500).json({ error: err.message });
  }
}