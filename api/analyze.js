/**
 * AgriCure - Smart API Handler
 * 
 * Priority:
 * 1. Local ML Model (Python Flask server) — if ML_SERVER_URL is set
 * 2. Gemini AI — if GEMINI_API_KEY is set
 * 3. Demo mode — fallback
 */

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { image, mimeType } = req.body;
    if (!image) return res.status(400).json({ error: 'No image provided' });

    // ── OPTION 1: Use local ML model server ──
    const mlServerUrl = process.env.ML_SERVER_URL;
    if (mlServerUrl) {
      try {
        // Wake up the server first
        await fetch(`${mlServerUrl}/health`, {
          signal: AbortSignal.timeout(30000)
        }).catch(() => {});
        
        const mlResponse = await fetch(`${mlServerUrl}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
          signal: AbortSignal.timeout(55000)
        });
        
        if (mlResponse.ok) {
          const result = await mlResponse.json();
          return res.status(200).json({ ...result, source: 'ml_model' });
        }
      } catch (mlErr) {
        console.warn('ML server error, falling back to Gemini:', mlErr.message);
      }
    }

    // ── OPTION 2: Use Gemini AI ──
    const geminiKey = process.env.GEMINI_API_KEY;
    if (geminiKey) {
      const prompt = `You are an expert agricultural plant pathologist.
Analyze this image and respond ONLY in this exact JSON format, no markdown, no extra text:
{"isPlant":true,"disease":"<disease name or Healthy Crop>","crop":"<crop type>","severity":<0-100>,"confidence":<0-100>,"description":"<2-3 sentences about what you see>","treatments":["<treatment 1>","<treatment 2>","<treatment 3>"],"prevention":"<one prevention tip>","notPlantMessage":"<fill only if not a plant>"}`;

      const geminiResponse = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${geminiKey}`,
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
            generationConfig: { temperature: 0.1, maxOutputTokens: 600 }
          })
        }
      );

      const rawText = await geminiResponse.text();
      
      let geminiData;
      try {
        geminiData = JSON.parse(rawText);
      } catch(e) {
        console.error('Gemini parse error:', rawText.substring(0, 200));
        // Fall through to demo mode
      }

      if (geminiData && !geminiData.error) {
        const raw = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || '';
        const cleaned = raw.replace(/```json|```/g, '').trim();
        try {
          const result = JSON.parse(cleaned);
          return res.status(200).json({ ...result, source: 'gemini' });
        } catch(e) {
          console.error('Result parse error:', cleaned.substring(0, 100));
        }
      }
    }

    // ── OPTION 3: Demo fallback ──
    const demoResults = [
      {
        isPlant: true,
        disease: "Leaf Rust (Demo Mode)",
        crop: "Wheat",
        severity: 72,
        confidence: 85,
        description: "Orange-brown pustules visible on upper leaf surfaces indicating wheat leaf rust infection at moderate severity. This is a demo result — connect your ML model or Gemini API for real analysis.",
        treatments: [
          "Apply propiconazole fungicide at 1ml/litre water",
          "Remove and destroy severely infected leaves",
          "Improve field drainage to reduce humidity"
        ],
        prevention: "Plant rust-resistant wheat varieties. Monitor weekly during humid weather conditions.",
        notPlantMessage: "",
        source: "demo"
      }
    ];

    return res.status(200).json(demoResults[0]);

  } catch (err) {
    console.error('Handler error:', err);
    return res.status(500).json({ error: err.message });
  }
}
