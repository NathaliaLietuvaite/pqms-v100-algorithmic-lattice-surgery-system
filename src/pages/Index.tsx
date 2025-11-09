import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlgorithmInput, OptimizationParams } from "@/components/AlgorithmInput";
import { OptimizedOutput } from "@/components/OptimizedOutput";
import ChatBot from "@/components/ChatBot";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { Sparkles, Cpu, LogOut, Shield } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

const Index = () => {
  const [optimizedResult, setOptimizedResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();
  const { user, session, loading, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    toast({
      title: "Abgemeldet",
      description: "Du wurdest erfolgreich abgemeldet.",
    });
  };

  const handleOptimize = async (algorithm: string, params: OptimizationParams) => {
    if (!user || !session) {
      toast({
        title: "Authentication Required",
        description: "Please sign in to use the PQMS V100 Resonance Engine.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setOptimizedResult(null);

    try {
      const { data, error } = await supabase.functions.invoke('optimize-algorithm', {
        body: { algorithm, params }
      });

      if (error) {
        console.error("Edge function error:", error);
        const errorMsg = error.message || JSON.stringify(error);
        if (errorMsg.includes("AI credits exhausted") || errorMsg.includes("credits") || errorMsg.includes("402")) {
          toast({
            title: "❌ AI Credits Exhausted",
            description: "Your Lovable AI credits are exhausted. Go to: Settings → Workspace → Usage → Add Credits",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(error.message || "Edge Function error");
      }

      if (data?.error) {
        console.error("API error:", data.error);
        if (data.error.includes("AI credits exhausted") || data.error.includes("credits")) {
          toast({
            title: "❌ AI Credits Exhausted",
            description: "Please add credits: Settings → Workspace → Usage → Add Credits",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(data.error);
      }

      if (data) {
        setOptimizedResult(data);
        toast({
          title: "✓ Optimization Complete",
          description: "Algorithm optimized via PQMS V100 Resonance Engine.",
        });
      } else {
        throw new Error("No optimization result received");
      }
    } catch (error) {
      console.error("Error optimizing algorithm:", error);
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (!errorMsg.includes("AI credits exhausted") && !errorMsg.includes("credits")) {
        toast({
          title: "Optimization Failed",
          description: errorMsg || "Failed to optimize. Please try again.",
          variant: "destructive",
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex items-center justify-center">
        <div className="text-center">
          <Sparkles className="h-12 w-12 text-primary animate-pulse mx-auto mb-4" />
          <p className="text-muted-foreground">Lädt...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm bg-card/30">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Sparkles className="h-8 w-8 text-primary" />
              </div>
              <div>
                <h1 className="text-3xl font-bold tracking-tight">PQMS V100 Algorithmic Lattice Surgery System</h1>
                <p className="text-muted-foreground text-sm">
                  Femtosecond Resonance Engine for Ethically-Gated Quantum Optimization
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <ChatBot />
              {user ? (
                <Button variant="outline" size="sm" onClick={handleSignOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Abmelden
                </Button>
              ) : (
                <Button variant="outline" size="sm" onClick={() => navigate("/auth")}>
                  Anmelden
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="space-y-12">
          {!user && (
            <Alert className="max-w-5xl mx-auto bg-primary/5 border-primary/20">
              <Shield className="h-4 w-4" />
              <AlertDescription>
                <strong>🔒 Authentication Required:</strong> Sign in with your free Lovable account to use the PQMS V100 Resonance Engine. 
                This system optimizes Algorithmic Lattice Surgery using femtosecond quantum resonance and ethical validation. 
                AI usage is billed to your Lovable account.
              </AlertDescription>
            </Alert>
          )}
          
          {/* Introduction */}
          <section className="max-w-5xl mx-auto text-center space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-semibold">
              <Cpu className="h-4 w-4" />
              Powered by PQMS V100 Resonance Engine
            </div>
            <h2 className="text-4xl font-bold tracking-tight">
              Transcending 4D Optimization via Physical Resonance
            </h2>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              The PQMS V100 Algorithmic Lattice Surgery System replaces classical computational optimization with physical resonance. 
              By treating ALS as a physical system seeking its ground state, we achieve sub-femtosecond operation latency while 
              embedding ethical governance directly into the quantum fabric through the Causal Ethics Cascade (CEK).
            </p>
          </section>

          {/* Algorithm Input */}
          <AlgorithmInput onOptimize={handleOptimize} isLoading={isLoading} />

          {/* Optimized Output */}
          {optimizedResult && (
            <OptimizedOutput
              optimizedCode={optimizedResult.optimizedCode}
              verilogCode={optimizedResult.verilogCode}
              qutipSim={optimizedResult.qutipSim}
              synthesisReport={optimizedResult.synthesisReport}
              metrics={optimizedResult.metrics}
              paper={optimizedResult.paper}
            />
          )}

          {/* Framework Info */}
          {!optimizedResult && !isLoading && (
            <section className="max-w-5xl mx-auto backdrop-blur-sm bg-card/20 p-8 rounded-xl border border-border/30">
              <h3 className="text-xl font-semibold mb-4">About the PQMS V100 Algorithmic Lattice Surgery System</h3>
              <div className="grid gap-4 text-sm text-muted-foreground">
                <p>
                  This system transcends conventional 4D spacetime constraints by replacing computational optimization 
                  with physical resonance. The Proactive Resonance Manifold (PRM) and Wormhole-like Synergies determine 
                  optimal lattice surgery paths as ground states of an ethical Hamiltonian at femtosecond timescales.
                </p>
                <div className="grid sm:grid-cols-2 gap-3 mt-4">
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Resonance Engine</div>
                    <div className="text-xs">PRM • Wormhole Synergies • Sub-femtosecond operation</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Ethical Validation</div>
                    <div className="text-xs">CEK • RCF &gt;0.999 • Confidence &gt;0.98</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Hardware Ready</div>
                    <div className="text-xs">Alveo U250 • Photonic Cube • YbB Dual-State</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Complete Output</div>
                    <div className="text-xs">Python • Verilog • QuTiP • Synthesis • Paper</div>
                  </div>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-20">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center text-sm text-muted-foreground space-y-2">
            <p className="font-semibold text-foreground">
              PQMS V100 Algorithmic Lattice Surgery System
            </p>
            <p>
              Developed by Nathalia Lietuvaite • Framework: PQMS v100 • License: MIT
            </p>
            <p className="text-xs">
              "Transcending 4D Optimization via Femtosecond Resonance" • CEK Ethics • Physical Ground States
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
