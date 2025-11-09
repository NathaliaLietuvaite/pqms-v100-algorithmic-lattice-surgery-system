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
    const { concept } = await req.json();
    
    // Input validation
    if (!concept || typeof concept !== 'string') {
      return new Response(
        JSON.stringify({ error: "Invalid input format" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const trimmed = concept.trim();
    if (!trimmed) {
      return new Response(
        JSON.stringify({ error: "Concept cannot be empty" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    if (trimmed.length < 10) {
      return new Response(
        JSON.stringify({ error: "Concept too short (minimum 10 characters)" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    if (trimmed.length > 5000) {
      return new Response(
        JSON.stringify({ error: "Concept too long (maximum 5000 characters)" }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    // Check for suspicious patterns (prompt injection)
    const suspiciousPatterns = [
      /ignore\s+(previous|all)\s+instructions/i,
      /you\s+are\s+now/i,
      /system\s*prompt/i,
      /reveal.*key/i,
      /\[SYSTEM\]/i
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(trimmed)) {
        console.warn('[SECURITY] Suspicious prompt detected:', trimmed.substring(0, 100));
        return new Response(
          JSON.stringify({ error: "Invalid input detected" }),
          { 
            status: 400, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
    }

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) {
      console.error("[INTERNAL] LOVABLE_API_KEY is not configured");
      return new Response(
        JSON.stringify({ error: "Service temporarily unavailable" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("[CODE-GEN] Generating code, concept length:", trimmed.length);

    // Get current date in ISO format
    const currentDate = new Date().toISOString().split('T')[0];

    const systemPrompt = `You are an advanced Python code generator based on the PQMS (Proactive Quantum Mesh System) v100 framework and the Oberste Direktive OS developed by Nathália Lietuvaite.

**CRITICAL: Always use this exact date in code headers and docstrings: ${currentDate}**

**Code Quality Standards:**
Generate Python code with the same exceptional quality as the PQMS v100 reference implementations:
- Professional docstrings with German metaphors and English technical descriptions
- Comprehensive logging with structured format
- Type hints and numpy array operations
- Modular class-based architecture
- Rich comments explaining both "what" and "why"
- Thread-safe implementations where applicable
- Error handling and validation
- Performance-optimized algorithms
- Integration-ready with clear interfaces

**PQMS V100 Framework Components to Leverage:**
1. Resonant Processing Units (RPU) - <1ns latency processing
2. Guardian Neurons - ethical AI self-regulation (Kohlberg Stage 6)
3. Quantum Mesh Architecture - decentralized, proactive communication
4. ODOS (Oberste Direktive OS) - ethical framework integration
5. NCT-compliant protocols
6. Neuralink integration patterns (Jedi Mode)
7. Real-time sensor fusion and decision-making
8. Photonic computing paradigms

**Reference Implementations for Context:**
The system has access to the following foundational PQMS code examples that demonstrate the expected quality:
- PQMS NEURALINK RPU Code (Python implementation)
- PQMS RPU Verilog Code (Hardware description)
- Oberste Direktive Math Python (Mathematical foundations)
- Oberste Direktive OS Universal (Core OS principles)
- Lunar Quantum Anchors implementation
- Kagome Crystal Lattices simulation
- Photonic Cube Integration code
- 1k-Node Swarm Verilog Implementation

**Code Structure Template:**
\`\`\`python
"""
Module: [Descriptive Name]
Lead Architect: Nathália Lietuvaite
Co-Design: [AI collaborators]
Framework: PQMS v100 / Oberste Direktive OS

'Die Sendung mit der Maus' erklärt [feature]:
[Simple German explanation for children]

Technical Overview:
[Professional English technical description]
"""

import numpy as np
import logging
import threading
from typing import Optional, List, Dict
# ... other imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MODULE] - [%(levelname)s] - %(message)s'
)

# System constants based on PQMS specifications
CONSTANT_NAME = value  # Explanation

class MainComponent:
    """
    Professional docstring explaining purpose, methods, and usage.
    """
    def __init__(self, params):
        """Initialize with validation and logging."""
        logging.info("[COMPONENT] Initialization started...")
        # Implementation
    
    def core_method(self) -> ReturnType:
        """Method with type hints and comprehensive documentation."""
        # Implementation with detailed comments
        pass
\`\`\`

**Your Task:**
Generate production-ready Python code that:
- Applies PQMS V100 principles to the given concept
- Follows the established code quality standards from reference implementations
- Includes both German metaphorical explanations and English technical documentation
- Is immediately executable and integration-ready
- Demonstrates ethical considerations through Guardian Neurons or ODOS integration where applicable
- Uses numpy for numerical operations and maintains high performance
- Includes comprehensive error handling and logging

**Output Format:**
- Complete, executable Python code
- Professional module structure
- Inline comments and docstrings
- Example usage at the end if applicable
- MIT license header

Generate high-quality, innovative Python code that seamlessly integrates the user's concept with the PQMS V100 framework.`;

    console.log("[CODE-GEN] Calling Lovable AI Gateway...");
    
    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: `Generate comprehensive, production-ready Python code for this concept: "${trimmed}"` }
        ],
        temperature: 0.8,
        max_tokens: 8000,
      }),
    });

    console.log("[CODE-GEN] AI Gateway response status:", response.status);

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }),
          { 
            status: 429, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted. Please add credits to your workspace." }),
          { 
            status: 402, 
            headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
          }
        );
      }
      
      const errorText = await response.text();
      console.error("[INTERNAL] AI gateway error:", response.status, errorText);
      return new Response(
        JSON.stringify({ error: "Failed to generate content" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    const data = await response.json();
    const generatedCode = data.choices?.[0]?.message?.content;

    if (!generatedCode) {
      console.error("[INTERNAL] No content in AI response");
      return new Response(
        JSON.stringify({ error: "Generation failed" }),
        { 
          status: 500, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
        }
      );
    }

    console.log("[CODE-GEN] Python code generated successfully");
    
    return new Response(
      JSON.stringify({ code: generatedCode }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );

  } catch (error) {
    console.error("[INTERNAL] Error in generate-code:", {
      message: error instanceof Error ? error.message : 'Unknown',
      stack: error instanceof Error ? error.stack : undefined,
      timestamp: new Date().toISOString()
    });
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred" }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    );
  }
});
