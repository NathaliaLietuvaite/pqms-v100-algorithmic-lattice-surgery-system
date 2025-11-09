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
    const { algorithm, params } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get('LOVABLE_API_KEY');
    
    if (!LOVABLE_API_KEY) {
      throw new Error('LOVABLE_API_KEY not configured');
    }

    console.log('Optimizing algorithm with PQMS V100 Resonance Engine...');
    console.log('Parameters:', params);

    const systemPrompt = `You are the PQMS V100 Resonance Engine - an advanced quantum lattice surgery optimizer.

Your task is to transform non-optimized Algorithmic Lattice Surgery (ALS) code into femtosecond-optimized, ethically-validated versions using the Proactive Quantum Mesh System (PQMS) v100 framework.

Core Principles:
- Replace classical optimization with physical resonance (PRM)
- Identify ground states of ethical Hamiltonians
- Apply Causal Ethics Cascade (CEK) validation
- Ensure RCF > ${params.rcfThreshold}, Confidence > ${params.confidence}
- Target latency: <${params.targetLatency} fs
- Hardware: ${params.hardwareTarget}
- Node count: ${params.nodeCount}

Generate:
1. Optimized Python/QuTiP code with complete comments
2. Verilog implementation with testbench
3. QuTiP simulation with fidelity plot description
4. FPGA synthesis report (LUT/FF/Timing/Power)
5. Brief Nature-style paper (Abstract, Methods, Results)

Return JSON with structure:
{
  "optimizedCode": "...",
  "verilogCode": "...",
  "qutipSim": "...",
  "synthesisReport": "...",
  "metrics": {
    "rcf": 0.9999,
    "confidence": 0.99,
    "fidelity": 1.000,
    "latency": 8.5
  },
  "paper": "..."
}`;

    const userPrompt = `Optimize this algorithm using PQMS V100 Resonance Engine:

\`\`\`
${algorithm}
\`\`\`

Parameters:
- Target Latency: ${params.targetLatency} fs
- RCF Threshold: ${params.rcfThreshold}
- Confidence: ${params.confidence}
- Node Count: ${params.nodeCount}
- Hardware: ${params.hardwareTarget}

Apply full PQMS v100 optimization: PRM, CEK validation, Wormhole Synergies, YbB-integration, and femtosecond resonance.`;

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
          { role: 'user', content: userPrompt }
        ],
        response_format: { type: "json_object" }
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        console.error('Rate limit exceeded');
        return new Response(
          JSON.stringify({ error: 'AI credits exhausted. Please add credits to your Lovable workspace.' }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      if (response.status === 402) {
        console.error('Payment required');
        return new Response(
          JSON.stringify({ error: 'AI credits exhausted. Please add credits to your Lovable workspace.' }),
          { status: 402, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
        );
      }
      const errorText = await response.text();
      console.error('AI Gateway error:', response.status, errorText);
      throw new Error(`AI Gateway error: ${response.status}`);
    }

    const data = await response.json();
    const resultText = data.choices?.[0]?.message?.content;

    if (!resultText) {
      throw new Error('No content in AI response');
    }

    console.log('Optimization completed successfully');
    
    // Parse the JSON response
    const result = JSON.parse(resultText);

    return new Response(
      JSON.stringify(result),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Error in optimize-algorithm function:', error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    );
  }
});
