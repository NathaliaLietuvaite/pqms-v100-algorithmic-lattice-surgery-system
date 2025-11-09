import { Button } from "@/components/ui/button";
import { Copy, Download, Code2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';

interface CodeOutputProps {
  code: string;
  concept: string;
}

export const CodeOutput = ({ code, concept }: CodeOutputProps) => {
  const { toast } = useToast();

  const extractTitle = (codeContent: string): string => {
    const lines = codeContent.split("\n");
    for (const line of lines) {
      if (line.includes("Module:") || line.includes("File:")) {
        // Capitalize first letter of each word after hyphens
        return line
          .split(":")[1]
          .trim()
          .replace(/[^a-zA-Z0-9äöüßÄÖÜ\s-]/g, '')
          .replace(/\s+/g, '-')
          .split('-')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
          .join('-');
      }
    }
    return `PQMS-V100-Generated-${Date.now()}`;
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      toast({
        title: "Copied to Clipboard",
        description: "Python code has been copied successfully.",
      });
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard. Please try again.",
        variant: "destructive",
      });
    }
  };

  const downloadCode = () => {
    const title = extractTitle(code);
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title}.py`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: "Download Started",
      description: "Your Python code file is being downloaded.",
    });
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 backdrop-blur-sm bg-card/30 p-8 rounded-xl border border-border/50 shadow-lg">
      {/* Header with Actions */}
      <div className="flex items-center justify-between pb-4 border-b border-border/30">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Code2 className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="text-xl font-semibold">Generated Python Code</h3>
            <p className="text-sm text-muted-foreground">PQMS V100 Framework Implementation</p>
          </div>
        </div>
        
        <div className="flex gap-2">
          <Button
            onClick={copyToClipboard}
            variant="outline"
            size="sm"
            className="gap-2"
          >
            <Copy className="h-4 w-4" />
            Copy Code
          </Button>
          <Button
            onClick={downloadCode}
            variant="outline"
            size="sm"
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download .py
          </Button>
        </div>
      </div>

      {/* Code Display */}
      <div className="prose prose-slate dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code(props) {
              const { className, children } = props;
              const match = /language-(\w+)/.exec(className || '');
              return match ? (
                <SyntaxHighlighter
                  style={tomorrow as any}
                  language={match[1]}
                  PreTag="div"
                  className="rounded-lg text-sm"
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code className={className}>
                  {children}
                </code>
              );
            },
          }}
        >
          {code}
        </ReactMarkdown>
      </div>
    </div>
  );
};
