import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ConceptInput } from "@/components/ConceptInput";
import { PaperOutput } from "@/components/PaperOutput";
import { CodeOutput } from "@/components/CodeOutput";
import ChatBot from "@/components/ChatBot";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { Sparkles, BookOpen, Code2, LogOut, Shield } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { DEMO_PAPER, DEMO_CODE } from "@/data/demoContent";

const Index = () => {
  const [generatedPaper, setGeneratedPaper] = useState<string>("");
  const [generatedCode, setGeneratedCode] = useState<string>("");
  const [currentConcept, setCurrentConcept] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [testMode, setTestMode] = useState(false);
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

  const handleGeneratePaper = async (concept: string) => {
    if (testMode || !user || !session) {
      setGeneratedPaper(DEMO_PAPER);
      setCurrentConcept(testMode ? concept : "Demo-Beispiel");
      toast({
        title: testMode ? "🧪 Test-Modus" : "Demo-Modus",
        description: testMode ? "Demo-Daten werden verwendet (keine AI Credits verbraucht)" : "Melde dich an, um echte AI-generierte Inhalte zu erhalten.",
        variant: "default",
      });
      return;
    }

    setIsLoading(true);
    setGeneratedPaper("");
    setCurrentConcept(concept);

    try {
      const { data, error } = await supabase.functions.invoke('generate-paper', {
        body: { concept }
      });

      if (error) {
        console.error("Edge function error details:", error);
        // Check if it's a credits exhausted error
        const errorMsg = error.message || JSON.stringify(error);
        if (errorMsg.includes("AI credits exhausted") || errorMsg.includes("credits") || errorMsg.includes("402")) {
          toast({
            title: "❌ AI Credits Aufgebraucht",
            description: "Deine Lovable AI Credits sind aufgebraucht. Gehe zu: Einstellungen → Workspace → Usage → Credits hinzufügen",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(error.message || "Edge Function returned a non-2xx status code");
      }

      if (data?.error) {
        console.error("API error from edge function:", data.error);
        // Check for AI credits exhausted error
        if (data.error.includes("AI credits exhausted") || data.error.includes("credits")) {
          toast({
            title: "❌ AI Credits Aufgebraucht",
            description: "Deine Lovable AI Credits sind aufgebraucht. Bitte füge Credits hinzu unter: Einstellungen → Workspace → Usage → Credits hinzufügen",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(data.error);
      }

      if (data?.paper) {
        setGeneratedPaper(data.paper);
        toast({
          title: "Paper Generated Successfully",
          description: "Your V100 scientific paper has been generated.",
        });
      } else {
        throw new Error("No paper content received");
      }
    } catch (error) {
      console.error("Error generating paper:", error);
      // Don't show another error if we already showed the credits error
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (!errorMsg.includes("AI credits exhausted") && !errorMsg.includes("credits")) {
        toast({
          title: "Generation Failed",
          description: errorMsg || "Failed to generate paper. Please try again.",
          variant: "destructive",
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateCode = async (concept: string) => {
    if (testMode || !user || !session) {
      setGeneratedCode(DEMO_CODE);
      setCurrentConcept(testMode ? concept : "Demo-Beispiel");
      toast({
        title: testMode ? "🧪 Test-Modus" : "Demo-Modus",
        description: testMode ? "Demo-Daten werden verwendet (keine AI Credits verbraucht)" : "Melde dich an, um echte AI-generierte Inhalte zu erhalten.",
        variant: "default",
      });
      return;
    }

    setIsLoading(true);
    setGeneratedCode("");
    setCurrentConcept(concept);

    try {
      const { data, error } = await supabase.functions.invoke('generate-code', {
        body: { concept }
      });

      if (error) {
        console.error("Edge function error details:", error);
        // Check if it's a credits exhausted error
        const errorMsg = error.message || JSON.stringify(error);
        if (errorMsg.includes("AI credits exhausted") || errorMsg.includes("credits") || errorMsg.includes("402")) {
          toast({
            title: "❌ AI Credits Aufgebraucht",
            description: "Deine Lovable AI Credits sind aufgebraucht. Gehe zu: Einstellungen → Workspace → Usage → Credits hinzufügen",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(error.message || "Edge Function returned a non-2xx status code");
      }

      if (data?.error) {
        console.error("API error from edge function:", data.error);
        // Check for AI credits exhausted error
        if (data.error.includes("AI credits exhausted") || data.error.includes("credits")) {
          toast({
            title: "❌ AI Credits Aufgebraucht",
            description: "Deine Lovable AI Credits sind aufgebraucht. Bitte füge Credits hinzu unter: Einstellungen → Workspace → Usage → Credits hinzufügen",
            variant: "destructive",
            duration: 10000,
          });
          return;
        }
        throw new Error(data.error);
      }

      if (data?.code) {
        setGeneratedCode(data.code);
        toast({
          title: "Code Generated Successfully",
          description: "Your V100 Python code has been generated.",
        });
      } else {
        throw new Error("No code content received");
      }
    } catch (error) {
      console.error("Error generating code:", error);
      // Don't show another error if we already showed the credits error
      const errorMsg = error instanceof Error ? error.message : String(error);
      if (!errorMsg.includes("AI credits exhausted") && !errorMsg.includes("credits")) {
        toast({
          title: "Generation Failed",
          description: errorMsg || "Failed to generate code. Please try again.",
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
                <h1 className="text-3xl font-bold tracking-tight">PQMS V100 Innovation Generator</h1>
                <p className="text-muted-foreground text-sm">
                  Transforming Concepts into Comprehensive Scientific Papers
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {user && (
                <Button 
                  variant={testMode ? "default" : "outline"} 
                  size="sm" 
                  onClick={() => {
                    setTestMode(!testMode);
                    toast({
                      title: !testMode ? "🧪 Test-Modus Aktiviert" : "✅ Live-Modus Aktiviert",
                      description: !testMode ? "Demo-Daten werden verwendet - keine Credits verbraucht" : "Echte AI-Generierung ist aktiv",
                    });
                  }}
                >
                  {testMode ? "🧪 Test-Modus" : "Test-Modus"}
                </Button>
              )}
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
            <Alert className="max-w-4xl mx-auto bg-primary/5 border-primary/20">
              <Shield className="h-4 w-4" />
              <AlertDescription>
                <strong>🔒 Demo Mode:</strong> You're viewing example outputs. Sign in with your free Lovable account to generate custom AI-powered scientific papers and code using the full PQMS V100 Framework. This open-source system uses your Lovable credentials - no separate registration needed. Each user's AI usage is billed to their own Lovable account.
              </AlertDescription>
            </Alert>
          )}
          
          {/* Introduction */}
          <section className="max-w-4xl mx-auto text-center space-y-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-semibold">
              <BookOpen className="h-4 w-4" />
              Powered by PQMS V100 Framework
            </div>
            <h2 className="text-4xl font-bold tracking-tight">
              Generate Scientific Papers with V100 Intelligence
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Built on the revolutionary Proactive Quantum Mesh System (PQMS) v100 framework by Nathalia Lietuvaite, 
              this innovation generator transforms your concept ideas into comprehensive, academically rigorous scientific papers 
              that seamlessly integrate quantum computing, ethical AI, and cutting-edge technology principles.
            </p>
          </section>

          {/* Tabbed Interface for Paper and Code Generation */}
          <Tabs defaultValue="paper" className="w-full max-w-4xl mx-auto">
            <TabsList className="grid w-full grid-cols-2 mb-8">
              <TabsTrigger value="paper" className="gap-2">
                <BookOpen className="h-4 w-4" />
                Scientific Paper
              </TabsTrigger>
              <TabsTrigger value="code" className="gap-2">
                <Code2 className="h-4 w-4" />
                Python Code
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="paper" className="space-y-12">
              <ConceptInput 
                onGenerate={handleGeneratePaper} 
                isLoading={isLoading}
                buttonText="Generate V100 Scientific Paper"
                loadingText="Generating Scientific Paper..."
              />
              {generatedPaper && <PaperOutput paper={generatedPaper} concept={currentConcept} />}
            </TabsContent>
            
            <TabsContent value="code" className="space-y-12">
              <ConceptInput 
                onGenerate={handleGenerateCode} 
                isLoading={isLoading}
                buttonText="Generate V100 Python Code"
                loadingText="Generating Python Code..."
              />
              {generatedCode && <CodeOutput code={generatedCode} concept={currentConcept} />}
            </TabsContent>
          </Tabs>

          {/* Framework Info */}
          {!generatedPaper && !generatedCode && !isLoading && (
            <section className="max-w-4xl mx-auto backdrop-blur-sm bg-card/20 p-8 rounded-xl border border-border/30">
              <h3 className="text-xl font-semibold mb-4">About the PQMS V100 Framework</h3>
              <div className="grid gap-4 text-sm text-muted-foreground">
                <p>
                  The PQMS v100 is a revolutionary quantum-classical hybrid architecture achieving sub-nanosecond 
                  Earth-Mars communication through resonant co-processing, Guardian Neurons for ethical AI self-regulation, 
                  and photonic computing integration.
                </p>
                <div className="grid sm:grid-cols-2 gap-3 mt-4">
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Core Technology</div>
                    <div className="text-xs">RPU &lt;1ns latency • NCT-compliant • FPGA-ready</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Ethical Framework</div>
                    <div className="text-xs">Guardian Neurons • ODOS • Kohlberg Stage 6</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Performance</div>
                    <div className="text-xs">Fidelity 1.000 • 95% BW saving • 1-2 TOps/s</div>
                  </div>
                  <div className="p-4 rounded-lg bg-background/50 border border-border/30">
                    <div className="font-semibold text-foreground mb-1">Innovation</div>
                    <div className="text-xs">Photonic 5cm³ cube • RCF metrics • Light-based computing</div>
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
              PQMS V100 Innovation Generator
            </p>
            <p>
              Developed by Nathalia Lietuvaite • Framework: PQMS v100 • License: MIT
            </p>
            <p className="text-xs">
              "Ethik → Konzept → Generiertes System" • Resonance & Cooperative Intentionality
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
