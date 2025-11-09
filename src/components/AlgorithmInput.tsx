import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Sparkles, Loader2, Settings2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface AlgorithmInputProps {
  onOptimize: (algorithm: string, params: OptimizationParams) => void;
  isLoading: boolean;
}

export interface OptimizationParams {
  targetLatency: number;
  rcfThreshold: number;
  confidence: number;
  nodeCount: number;
  hardwareTarget: string;
}

export const AlgorithmInput = ({ onOptimize, isLoading }: AlgorithmInputProps) => {
  const [algorithm, setAlgorithm] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const { toast } = useToast();
  
  const [params, setParams] = useState<OptimizationParams>({
    targetLatency: 10,
    rcfThreshold: 0.999,
    confidence: 0.98,
    nodeCount: 1024,
    hardwareTarget: "alveo-u250"
  });

  const handleOptimize = () => {
    const trimmed = algorithm.trim();
    
    if (!trimmed) {
      toast({
        title: "Input Required",
        description: "Please enter an algorithm to optimize.",
        variant: "destructive",
      });
      return;
    }

    if (trimmed.length < 10) {
      toast({
        title: "Algorithm Too Short",
        description: "Please provide at least 10 characters for meaningful optimization.",
        variant: "destructive",
      });
      return;
    }

    onOptimize(trimmed, params);
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6 backdrop-blur-sm bg-card/30 p-8 rounded-xl border border-border/50 shadow-lg">
      <div className="space-y-2">
        <Label htmlFor="algorithm" className="text-sm font-semibold text-foreground uppercase tracking-wide">
          Non-Optimized Lattice Surgery Algorithm
        </Label>
        <Textarea
          id="algorithm"
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
          placeholder={`# Enter your non-optimized algorithm here (Python/QuTiP, Verilog, or pseudocode)\n# Example:\nimport qutip as qt\nH = qt.rand_herm(4)\npsi = qt.basis(4, 0)\nresult = qt.mesolve(H, psi, [0, 10])`}
          className="min-h-[200px] resize-none text-base bg-background/50 backdrop-blur-sm border-border/50 focus:border-primary/50 transition-all font-mono"
          disabled={isLoading}
          maxLength={10000}
        />
        <p className="text-xs text-muted-foreground text-right">
          {algorithm.length}/10000 characters
        </p>
      </div>

      <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
        <CollapsibleTrigger asChild>
          <Button variant="outline" className="w-full justify-between">
            <span className="flex items-center gap-2">
              <Settings2 className="h-4 w-4" />
              Advanced Parameters
            </span>
            <span className="text-xs text-muted-foreground">
              {showAdvanced ? "Hide" : "Show"}
            </span>
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="latency">Target Latency (fs)</Label>
              <Input
                id="latency"
                type="number"
                value={params.targetLatency}
                onChange={(e) => setParams({...params, targetLatency: parseFloat(e.target.value)})}
                min={1}
                max={1000}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rcf">RCF Threshold</Label>
              <Input
                id="rcf"
                type="number"
                value={params.rcfThreshold}
                onChange={(e) => setParams({...params, rcfThreshold: parseFloat(e.target.value)})}
                min={0.9}
                max={1.0}
                step={0.001}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="confidence">Confidence Threshold</Label>
              <Input
                id="confidence"
                type="number"
                value={params.confidence}
                onChange={(e) => setParams({...params, confidence: parseFloat(e.target.value)})}
                min={0.9}
                max={1.0}
                step={0.01}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="nodes">Node Count</Label>
              <Input
                id="nodes"
                type="number"
                value={params.nodeCount}
                onChange={(e) => setParams({...params, nodeCount: parseInt(e.target.value)})}
                min={64}
                max={10000}
                step={64}
              />
            </div>
            <div className="space-y-2 md:col-span-2">
              <Label htmlFor="hardware">Hardware Target</Label>
              <Select 
                value={params.hardwareTarget} 
                onValueChange={(val) => setParams({...params, hardwareTarget: val})}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="alveo-u250">Xilinx Alveo U250</SelectItem>
                  <SelectItem value="photonic-cube">5cm³ Photonic Cube</SelectItem>
                  <SelectItem value="ybb-material">YbB Dual-State Material</SelectItem>
                  <SelectItem value="neuralink-rpu">Neuralink RPU</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
      
      <Button
        onClick={handleOptimize}
        disabled={isLoading || !algorithm.trim()}
        className="w-full h-12 text-base font-semibold bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 transition-all duration-300 shadow-md hover:shadow-lg"
      >
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Resonance Engine Processing...
          </>
        ) : (
          <>
            <Sparkles className="mr-2 h-5 w-5" />
            Optimize via PQMS Resonance
          </>
        )}
      </Button>
    </div>
  );
};
