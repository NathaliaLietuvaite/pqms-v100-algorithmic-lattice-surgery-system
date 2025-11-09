import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, Send, Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from "react-router-dom";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const MAX_MESSAGE_LENGTH = 2000;
const MAX_CONVERSATION_MESSAGES = 50;

const ChatBot = () => {
  const { user, session } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: user 
        ? "Hallo! Ich bin Nathalia Lietuvaite. Ich kann dir alles über das PQMS V100 Framework und die verfügbaren Dokumente erzählen. Was möchtest du wissen?"
        : "Hallo! Ich bin ein freundlicher Assistent. Im Demo-Modus kann ich allgemeine Fragen beantworten. Für vollständigen Zugriff auf das PQMS V100 Framework melde dich bitte an."
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || isLoading) return;

    // Input validation
    if (trimmedInput.length > MAX_MESSAGE_LENGTH) {
      toast.error(`Nachricht zu lang (max ${MAX_MESSAGE_LENGTH} Zeichen)`, {
        description: `Aktuelle Länge: ${trimmedInput.length}`
      });
      return;
    }

    // Conversation length validation
    if (messages.length >= MAX_CONVERSATION_MESSAGES) {
      toast.error("Unterhaltung zu lang", {
        description: "Bitte starte einen neuen Chat.",
        action: {
          label: "Zurücksetzen",
          onClick: () => setMessages([{
            role: "assistant",
            content: user 
              ? "Hallo! Ich bin Nathalia Lietuvaite. Ich kann dir alles über das PQMS V100 Framework und die verfügbaren Dokumente erzählen. Was möchtest du wissen?"
              : "Hallo! Ich bin ein freundlicher Assistent. Im Demo-Modus kann ich allgemeine Fragen beantworten. Für vollständigen Zugriff auf das PQMS V100 Framework melde dich bitte an."
          }])
        }
      });
      return;
    }

    const userMessage: Message = { role: "user", content: trimmedInput };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Demo mode for unauthenticated users
    if (!user || !session) {
      setTimeout(() => {
        const demoResponses = [
          "Das ist eine interessante Frage! Im Demo-Modus kann ich dir nur allgemeine Informationen geben. Melde dich an, um Zugriff auf das vollständige PQMS V100 Framework mit detailliertem Expertenwissen zu erhalten.",
          "Gerne würde ich dir mehr dazu erzählen! Der vollständige Chatbot mit Zugriff auf alle PQMS-Dokumente ist nach der Anmeldung verfügbar. Dort kann ich dir alle technischen Details erklären.",
          "Das klingt spannend! Um dir präzise Antworten basierend auf dem PQMS V100 Framework zu geben, benötige ich den vollen Systemzugriff. Bitte melde dich an für das komplette Erlebnis."
        ];
        const randomResponse = demoResponses[Math.floor(Math.random() * demoResponses.length)];
        setMessages(prev => [...prev, {
          role: "assistant",
          content: randomResponse
        }]);
        setIsLoading(false);
        
        // Show login prompt after demo response
        toast.info("Demo-Modus aktiv", {
          description: "Melde dich an für vollständige KI-Antworten",
          action: {
            label: "Anmelden",
            onClick: () => {
              setOpen(false);
              navigate("/auth");
            }
          }
        });
      }, 1000);
      return;
    }

    // Full chatbot for authenticated users
    try {
      const { data, error } = await supabase.functions.invoke("chatbot", {
        body: { messages: [...messages, userMessage] }
      });

      if (error) throw error;

      if (data?.response) {
        setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
      }
    } catch (error) {
      console.error("Chatbot error:", error);
      toast.error("Fehler beim Verarbeiten der Nachricht");
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: "Entschuldigung, es gab einen Fehler bei der Verarbeitung deiner Anfrage." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="icon" className="relative">
          <MessageCircle className="h-5 w-5" />
          <span className="sr-only">Chat mit Nathalia</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px] h-[600px] flex flex-col">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white font-bold">
              NL
            </div>
            <DialogTitle>Nathalia Lietuvaite</DialogTitle>
          </div>
        </DialogHeader>
        
        <ScrollArea ref={scrollRef} className="flex-1 pr-4">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-muted rounded-lg p-3">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="flex gap-2 pt-4 border-t">
          <Input
            placeholder="Stelle eine Frage..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
          />
          <Button onClick={handleSend} disabled={isLoading || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ChatBot;
