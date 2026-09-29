import { useState } from "react";
import { Check, Copy, Shield, Terminal, X, AlertTriangle, Clock } from "lucide-react";
import { cn } from "@/lib/utils";
import type { CodeRun } from "@/services/types";

interface ExecutionResultProps {
  run: CodeRun;
  onClose?: () => void;
}

export function ExecutionResult({ run, onClose }: ExecutionResultProps) {
  const [copied, setCopied] = useState(false);
  const isSuccess = run.status === "succeeded";
  const isTimedOut = run.status === "timed_out";

  const handleCopy = () => {
    const text = [run.stdout, run.stderr].filter(Boolean).join("\n");
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="mt-3 rounded-lg border border-border/80 bg-background/95 overflow-hidden text-xs shadow-sm">
      {/* Header bar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/60 bg-surface/50">
        <div className="flex items-center gap-2">
          <Terminal className="size-3.5 text-primary" />
          <span className="font-medium text-foreground">Sandbox Execution</span>
          <span
            className={cn(
              "px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider font-semibold",
              isSuccess
                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                : isTimedOut
                ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
            )}
          >
            {run.status}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleCopy}
            className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-surface transition-colors"
            title="Copy output"
          >
            {copied ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
          </button>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-surface transition-colors"
              title="Dismiss output"
            >
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Governance & runtime badges */}
      <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 border-b border-border/40 bg-surface/20 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
          <Shield className="size-3" /> Policy approved
        </span>
        <span className="text-border">·</span>
        <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
          <Check className="size-3" /> Sandbox isolated
        </span>
        <span className="text-border">·</span>
        <span>Network disabled</span>
        {run.durationMs !== undefined && (
          <>
            <span className="text-border">·</span>
            <span className="inline-flex items-center gap-1 font-mono">
              <Clock className="size-3" /> {run.durationMs}ms
            </span>
          </>
        )}
        {run.exitCode !== undefined && (
          <>
            <span className="text-border">·</span>
            <span className="font-mono">exit code {run.exitCode}</span>
          </>
        )}
      </div>

      {/* Stdout Output */}
      {run.stdout && (
        <div className="p-3 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap text-emerald-200/90 bg-black/40">
          {run.stdout}
        </div>
      )}

      {/* Stderr Output */}
      {run.stderr && (
        <div className="p-3 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap text-rose-300 bg-rose-950/20 border-t border-rose-900/30">
          <div className="flex items-center gap-1 text-rose-400 mb-1 font-sans text-[10px] uppercase tracking-wider font-semibold">
            <AlertTriangle className="size-3" /> stderr
          </div>
          {run.stderr}
        </div>
      )}

      {/* Empty output state */}
      {!run.stdout && !run.stderr && (
        <div className="p-3 text-center text-muted-foreground italic text-[11px]">
          Program exited with no output.
        </div>
      )}
    </div>
  );
}
