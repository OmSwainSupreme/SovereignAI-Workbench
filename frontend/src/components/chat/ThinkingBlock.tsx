import { useEffect, useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThinkingBlockProps {
  thinking: string;
  isStreaming: boolean;
  hasContent: boolean;
}

export function ThinkingBlock({ thinking, isStreaming, hasContent }: ThinkingBlockProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [userInteracted, setUserInteracted] = useState(false);
  const [copied, setCopied] = useState(false);

  // Auto-manage open state: open while thinking before content arrives, then collapse
  useEffect(() => {
    if (!userInteracted) {
      if (isStreaming && !hasContent) {
        setIsOpen(true);
      } else if (hasContent) {
        setIsOpen(false);
      }
    }
  }, [isStreaming, hasContent, userInteracted]);

  if (!thinking || thinking.trim().length === 0) return null;

  const isActivelyThinking = isStreaming && !hasContent;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(thinking);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-2.5 max-w-full">
      <button
        type="button"
        onClick={() => {
          setUserInteracted(true);
          setIsOpen((prev) => !prev);
        }}
        className={cn(
          "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium transition-all cursor-pointer shadow-2xs",
          isActivelyThinking
            ? "border-primary/50 bg-primary/10 text-primary hover:bg-primary/20"
            : "border-border/60 bg-surface/60 text-muted-foreground hover:bg-surface hover:text-foreground"
        )}
        aria-expanded={isOpen}
      >
        <span className="relative flex size-2">
          {isActivelyThinking ? (
            <>
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full size-2 bg-primary" />
            </>
          ) : (
            <span className="inline-flex rounded-full size-2 bg-emerald-400" />
          )}
        </span>

        <span className="font-sans font-medium">
          {isActivelyThinking ? "Thinking..." : "Thought process"}
        </span>

        <span className="text-[10px] text-muted-foreground ml-0.5">
          {isOpen ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
        </span>
      </button>

      {isOpen && (
        <div className="relative mt-2.5 max-h-72 overflow-y-auto rounded-xl border border-border/70 bg-surface/40 p-3.5 pl-4 font-mono text-[11px] leading-relaxed text-muted-foreground/90 whitespace-pre-wrap selection:bg-primary/20 shadow-inner border-l-2 border-l-primary/60">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-border/40 font-sans text-[10px] text-muted-foreground uppercase tracking-wider">
            <span className="flex items-center gap-1.5 font-medium text-foreground/80">
              <Sparkles className="size-3 text-primary" />
              Internal reasoning trace
            </span>
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              title="Copy thinking trace"
            >
              {copied ? <Check className="size-3 text-emerald-400" /> : <Copy className="size-3" />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
          </div>
          {thinking}
          {isActivelyThinking && (
            <span className="ml-1 inline-block h-3.5 w-1.5 animate-pulse bg-primary align-text-bottom" />
          )}
        </div>
      )}
    </div>
  );
}
