import { Auth as SupabaseAuth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import { supabase } from "@/integrations/supabase/client";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Sparkles } from "lucide-react";

const Auth = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is already logged in
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) {
        navigate("/");
      }
    });

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_IN" && session) {
        navigate("/");
      }
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="p-2 rounded-lg bg-primary/10">
              <Sparkles className="h-8 w-8 text-primary" />
            </div>
            <h1 className="text-3xl font-bold tracking-tight">PQMS V100</h1>
          </div>
          <p className="text-muted-foreground">
            Melde dich an, um AI-gestützte Forschungspapiere und Code zu generieren
          </p>
        </div>
        
        <div className="backdrop-blur-sm bg-card/80 p-8 rounded-xl border border-border/30 shadow-lg">
          <SupabaseAuth
            supabaseClient={supabase}
            appearance={{
              theme: ThemeSupa,
              variables: {
                default: {
                  colors: {
                    brand: "hsl(var(--primary))",
                    brandAccent: "hsl(var(--primary))",
                  },
                },
              },
            }}
            localization={{
              variables: {
                sign_in: {
                  email_label: "E-Mail",
                  password_label: "Passwort",
                  email_input_placeholder: "Deine E-Mail-Adresse",
                  password_input_placeholder: "Dein Passwort",
                  button_label: "Anmelden",
                  loading_button_label: "Anmeldung läuft...",
                  social_provider_text: "Mit {{provider}} anmelden",
                  link_text: "Hast du bereits ein Konto? Melde dich an",
                },
                sign_up: {
                  email_label: "E-Mail",
                  password_label: "Passwort",
                  email_input_placeholder: "Deine E-Mail-Adresse",
                  password_input_placeholder: "Dein Passwort",
                  button_label: "Registrieren",
                  loading_button_label: "Registrierung läuft...",
                  social_provider_text: "Mit {{provider}} registrieren",
                  link_text: "Noch kein Konto? Registriere dich",
                },
              },
            }}
            providers={[]}
            redirectTo={window.location.origin}
          />
        </div>
        
        <div className="mt-6 text-center text-sm text-muted-foreground">
          <p>
            Durch die Registrierung stimmst du den Nutzungsbedingungen zu.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
