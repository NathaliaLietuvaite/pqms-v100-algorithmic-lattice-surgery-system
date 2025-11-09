import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Sparkles, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ConceptInputProps {
  onGenerate: (concept: string) => void;
  isLoading: boolean;
  buttonText?: string;
  loadingText?: string;
}

export const ConceptInput = ({ onGenerate, isLoading, buttonText = "Generate", loadingText = "Generating..." }: ConceptInputProps) => {
  const [concept, setConcept] = useState("");
  const { toast } = useToast();

  const handleGenerate = () => {
    const trimmed = concept.trim();
    
    if (!trimmed) {
      toast({
        title: "Eingabe erforderlich",
        description: "Bitte gib ein Konzept ein, um ein wissenschaftliches Paper zu generieren.",
        variant: "destructive",
      });
      return;
    }

    if (trimmed.length < 10) {
      toast({
        title: "Konzept zu kurz",
        description: "Bitte gib mindestens 10 Zeichen ein für eine aussagekräftige Generierung.",
        variant: "destructive",
      });
      return;
    }

    if (trimmed.length > 5000) {
      toast({
        title: "Konzept zu lang",
        description: `Maximal 5000 Zeichen erlaubt. Aktuelle Länge: ${trimmed.length}`,
        variant: "destructive",
      });
      return;
    }

    onGenerate(trimmed);
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 backdrop-blur-sm bg-card/30 p-8 rounded-xl border border-border/50 shadow-lg">
      <div className="space-y-2">
        <label htmlFor="concept" className="text-sm font-semibold text-foreground uppercase tracking-wide">
          Concept Input
        </label>
        <Textarea
          id="concept"
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          placeholder="Enter your concept idea here... (e.g., 'Integration of quantum computing with neural networks for real-time consciousness mapping')"
          className="min-h-[150px] resize-none text-base bg-background/50 backdrop-blur-sm border-border/50 focus:border-primary/50 transition-all"
          disabled={isLoading}
          maxLength={5000}
        />
        <p className="text-xs text-muted-foreground text-right">
          {concept.length}/5000 Zeichen
        </p>
      </div>
      
      <Button
        onClick={handleGenerate}
        disabled={isLoading || !concept.trim()}
        className="w-full h-12 text-base font-semibold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 transition-all duration-300 shadow-md hover:shadow-lg"
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            {loadingText}
          </>
        ) : (
          <>
            <Sparkles className="mr-2 h-5 w-5" />
            {buttonText}
          </>
        )}
      </Button>
    </div>
  );
};
