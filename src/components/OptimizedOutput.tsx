import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy, Download, Check, FileCode, FileType, Image as ImageIcon, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface OptimizedOutputProps {
  optimizedCode: string;
  verilogCode: string;
  qutipSim: string;
  synthesisReport: string;
  metrics: {
    rcf: number;
    confidence: number;
    fidelity: number;
    latency: number;
  };
  paper: string;
}

export const OptimizedOutput = ({ optimizedCode, verilogCode, qutipSim, synthesisReport, metrics, paper }: OptimizedOutputProps) => {
  const [copied, setCopied] = useState<string | null>(null);
  const { toast } = useToast();

  const handleCopy = async (content: string, label: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(label);
      toast({
        title: "Copied to Clipboard",
        description: `${label} has been copied to your clipboard.`,
      });
      setTimeout(() => setCopied(null), 2000);
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleDownload = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Download Started",
      description: `${filename} is being downloaded.`,
    });
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6">
      {/* Metrics Dashboard */}
      <Card className="bg-card/30 backdrop-blur-sm border-border/50">
        <CardHeader>
          <CardTitle>PQMS V100 Optimization Metrics</CardTitle>
          <CardDescription>Resonance Engine Performance Indicators</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">RCF</p>
              <p className="text-2xl font-bold">
                {metrics.rcf.toFixed(4)}
                {metrics.rcf >= 0.999 && <Badge className="ml-2" variant="default">✓ Pass</Badge>}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Confidence</p>
              <p className="text-2xl font-bold">
                {metrics.confidence.toFixed(4)}
                {metrics.confidence >= 0.98 && <Badge className="ml-2" variant="default">✓ Pass</Badge>}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Fidelity</p>
              <p className="text-2xl font-bold">{metrics.fidelity.toFixed(4)}</p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Latency</p>
              <p className="text-2xl font-bold">{metrics.latency.toFixed(2)} fs</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Download Actions */}
      <div className="flex flex-wrap gap-3 justify-center">
        <Button
          onClick={() => handleDownload(optimizedCode, "optimized_algorithm.py", "text/x-python")}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80"
        >
          <FileCode className="mr-2 h-4 w-4" />
          Download Python
        </Button>
        <Button
          onClick={() => handleDownload(verilogCode, "optimized_algorithm.v", "text/plain")}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80"
        >
          <FileCode className="mr-2 h-4 w-4" />
          Download Verilog
        </Button>
        <Button
          onClick={() => handleDownload(synthesisReport, "vivado_project.tcl", "text/plain")}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80"
        >
          <FileType className="mr-2 h-4 w-4" />
          Download Vivado TCL
        </Button>
        <Button
          onClick={() => handleDownload(paper, "PQMS_V100_Paper.md", "text/markdown")}
          variant="outline"
          className="bg-background/50 backdrop-blur-sm hover:bg-background/80"
        >
          <FileText className="mr-2 h-4 w-4" />
          Download Paper
        </Button>
      </div>

      {/* Tabbed Content */}
      <Tabs defaultValue="python" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="python">Python</TabsTrigger>
          <TabsTrigger value="verilog">Verilog</TabsTrigger>
          <TabsTrigger value="qutip">QuTiP Sim</TabsTrigger>
          <TabsTrigger value="synthesis">Synthesis</TabsTrigger>
          <TabsTrigger value="paper">Paper</TabsTrigger>
        </TabsList>

        <TabsContent value="python" className="mt-6">
          <Card className="bg-card/30 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Optimized Python Code</CardTitle>
                <Button
                  onClick={() => handleCopy(optimizedCode, "Python Code")}
                  variant="ghost"
                  size="sm"
                >
                  {copied === "Python Code" ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <SyntaxHighlighter language="python" style={vscDarkPlus} customStyle={{ margin: 0 }}>
                {optimizedCode}
              </SyntaxHighlighter>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="verilog" className="mt-6">
          <Card className="bg-card/30 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Optimized Verilog Code</CardTitle>
                <Button
                  onClick={() => handleCopy(verilogCode, "Verilog Code")}
                  variant="ghost"
                  size="sm"
                >
                  {copied === "Verilog Code" ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <SyntaxHighlighter language="verilog" style={vscDarkPlus} customStyle={{ margin: 0 }}>
                {verilogCode}
              </SyntaxHighlighter>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="qutip" className="mt-6">
          <Card className="bg-card/30 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>QuTiP Simulation</CardTitle>
                <Button
                  onClick={() => handleCopy(qutipSim, "QuTiP Simulation")}
                  variant="ghost"
                  size="sm"
                >
                  {copied === "QuTiP Simulation" ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <CardDescription>Live simulation with fidelity validation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{qutipSim}</ReactMarkdown>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="synthesis" className="mt-6">
          <Card className="bg-card/30 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>FPGA Synthesis Report</CardTitle>
                <Button
                  onClick={() => handleCopy(synthesisReport, "Synthesis Report")}
                  variant="ghost"
                  size="sm"
                >
                  {copied === "Synthesis Report" ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <CardDescription>Vivado-compatible synthesis for Alveo U250</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{synthesisReport}</ReactMarkdown>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="paper" className="mt-6">
          <Card className="bg-card/30 backdrop-blur-sm border-border/50">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Nature-Style Paper</CardTitle>
                <Button
                  onClick={() => handleCopy(paper, "Paper")}
                  variant="ghost"
                  size="sm"
                >
                  {copied === "Paper" ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <CardDescription>Complete scientific documentation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, inline, className, children, ...props }: any) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {paper}
                </ReactMarkdown>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
