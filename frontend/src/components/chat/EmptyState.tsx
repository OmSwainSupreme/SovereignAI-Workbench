import { ShieldCheck, FileText, Terminal, Image, Files } from "lucide-react";

interface EmptyStateProps {
  onSelectPrompt: (prompt: string) => void;
}

const SUGGESTED_PROMPTS = [
  {
    icon: FileText,
    title: "Analyze this document",
    description: "Extract insights, key terms, and summaries from reports",
    prompt: "Summarize this document and extract the top 3 key takeaways.",
  },
  {
    icon: Terminal,
    title: "Write and run code",
    description: "Generate and test algorithms in isolated Docker sandbox",
    prompt: "Write a Python function to check whether a number is prime and test it with sample inputs.",
  },
  {
    icon: Image,
    title: "Analyze an image",
    description: "Run local OCR or multimodal scene analysis",
    prompt: "Analyze this image and describe its key visual elements and any visible text.",
  },
  {
    icon: Files,
    title: "Summarize my files",
    description: "Inspect workspace documents and produce structured notes",
    prompt: "Inspect the files in the workspace and give me a high-level overview of the project.",
  },
];

export function EmptyState({ onSelectPrompt }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] max-w-2xl mx-auto px-4 text-center">
      {/* SovereignAI Crest */}
      <div className="relative mb-6 flex size-14 items-center justify-center rounded-2xl bg-gradient-to-b from-primary/25 to-primary/10 border border-primary/30 shadow-lg shadow-primary/5">
        <ShieldCheck className="size-8 text-primary" />
        <span className="absolute -bottom-1 -right-1 size-3.5 rounded-full bg-emerald-500 border-2 border-background" />
      </div>

      <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-foreground font-sans">
        Private intelligence.
        <br />
        <span className="text-muted-foreground font-normal">Built for your environment.</span>
      </h1>

      <p className="mt-3 text-sm text-muted-foreground max-w-md leading-relaxed">
        Reason, create, analyze and automate with AI that stays entirely on your infrastructure.
      </p>

      {/* Suggested prompts */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
        {SUGGESTED_PROMPTS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.title}
              type="button"
              onClick={() => onSelectPrompt(item.prompt)}
              className="group p-3.5 rounded-xl border border-border/70 bg-surface/40 hover:bg-surface/90 hover:border-primary/40 transition-all text-left flex items-start gap-3 shadow-xs hover:shadow-sm"
            >
              <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-surface border border-border/60 text-muted-foreground group-hover:text-primary group-hover:border-primary/30 transition-colors">
                <Icon className="size-4" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-foreground group-hover:text-primary transition-colors">
                  {item.title}
                </div>
                <div className="text-[11px] text-muted-foreground mt-0.5 line-clamp-1">
                  {item.description}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
