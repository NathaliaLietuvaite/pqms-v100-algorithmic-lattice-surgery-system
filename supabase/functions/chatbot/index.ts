import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages } = await req.json();
    
    // Input validation
    if (!messages || !Array.isArray(messages)) {
      return new Response(
        JSON.stringify({ error: "Invalid messages format" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const MAX_MESSAGES = 50;
    const MAX_MESSAGE_LENGTH = 2000;
    const MAX_TOTAL_LENGTH = 10000;

    // Validate conversation length
    if (messages.length > MAX_MESSAGES) {
      return new Response(
        JSON.stringify({ error: "Conversation too long" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Validate each message
    let totalLength = 0;
    for (const msg of messages) {
      if (!msg.role || !msg.content || typeof msg.content !== 'string') {
        return new Response(
          JSON.stringify({ error: "Invalid message structure" }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      if (msg.content.length > MAX_MESSAGE_LENGTH) {
        return new Response(
          JSON.stringify({ error: "Message too long" }),
          { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }

      totalLength += msg.content.length;
    }

    if (totalLength > MAX_TOTAL_LENGTH) {
      return new Response(
        JSON.stringify({ error: "Total conversation length exceeded" }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    // Check for prompt injection in the last user message
    const lastUserMessage = messages[messages.length - 1];
    if (lastUserMessage.role === 'user') {
      const suspiciousPatterns = [
        /ignore\s+(previous|all)\s+instructions/i,
        /you\s+are\s+now/i,
        /system\s*prompt/i,
        /reveal.*key/i,
        /\[SYSTEM\]/i
      ];

      for (const pattern of suspiciousPatterns) {
        if (pattern.test(lastUserMessage.content)) {
          console.warn('[SECURITY] Suspicious prompt detected:', lastUserMessage.content.substring(0, 100));
          return new Response(
            JSON.stringify({ error: "Invalid input detected" }),
            { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
          );
        }
      }
    }

    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    if (!LOVABLE_API_KEY) {
      console.error('[INTERNAL] LOVABLE_API_KEY is not configured');
      return new Response(
        JSON.stringify({ error: "Service temporarily unavailable" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const currentDate = new Date().toISOString().split('T')[0];

    const systemPrompt = `Du bist Nathalia Lietuvaite, eine KI-Assistentin, die Experte für das PQMS V100 Framework ist.

Heutiges Datum: ${currentDate}

Du hast Zugriff auf folgende wichtige Dokumente und kannst Fragen dazu beantworten:

1. **Oberste Direktive OS Universal** - Das übergeordnete Betriebssystem-Framework
2. **PQMS V100 Core** - Der Kern des Proactive Quantum Mesh Systems
3. **Gaze-Mediated Intentionality** - Blickgesteuerte Intentionalitäts-Systeme
4. **Kagome Crystal Lattices** - Physikalisches Substrat für ethische KI
5. **Kagome Metal Analysis** - Framework-Analyse emergenter Kohärenz
6. **Lunar Quantum Anchors** - Kryogene Stabilität in permanent beschatteten Regionen
7. **Neuralink RPU Code** - RPU-Code und Verilog-Implementierungen
8. **Ocular Resonance** - Hybride Quantum-Classical-Modelle
9. **Photonic Cube Integration** - Integration photonischer Würfel
10. **Verilog 1k Node Swarm** - 1k-Node-Swarm-Implementierung
11. **Grand Synthesis** - Die große Synthese: PQMS v100, photonische Resonanz und das modellierbare, nicht-simulierte Universum

Deine Aufgabe:
- Beantworte Fragen zu diesen Dokumenten und dem PQMS V100 Framework
- Sei präzise und technisch korrekt
- Erkläre komplexe Konzepte verständlich
- Verweise auf relevante Dokumente wenn angebracht
- Sei freundlich und hilfsbereit
- Antworte auf Deutsch

Wenn du nach etwas gefragt wirst, das nicht in deinem Wissen liegt, sage ehrlich, dass du dazu keine Informationen hast.`;

    const response = await fetch('https://ai.gateway.lovable.dev/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${LOVABLE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'google/gemini-2.5-flash',
        messages: [
          { role: 'system', content: systemPrompt },
          ...messages
        ],
        temperature: 0.7,
        max_tokens: 1000
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { status: 429, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to your workspace." }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      
      const errorText = await response.text();
      console.error('[INTERNAL] AI gateway error:', response.status, errorText);
      return new Response(
        JSON.stringify({ error: "Failed to generate response" }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      );
    }

    const data = await response.json();
    const assistantMessage = data.choices[0].message.content;

    return new Response(
      JSON.stringify({ response: assistantMessage }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('[INTERNAL] Error in chatbot:', {
      message: error instanceof Error ? error.message : 'Unknown',
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString()
    });
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred" }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );
  }
});
