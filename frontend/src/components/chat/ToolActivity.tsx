import { useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronUp,
  FileSearch,
  FileText,
  Image,
  Loader2,
  ShieldCheck,
  Terminal,
  Wrench,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolEvent, ToolStatus } from "@/services/types";

interface ToolActivityProps {
  tools: ToolEvent[];
}

function getToolIcon(label: string) {
  const lower = label.toLowerCase();
  if (lower.includes("policy") || lower.includes("security")) return ShieldCheck;
  if (lower.includes("code") || lower.includes("sandbox")) return Terminal;
  if (lower.includes("vision") || lower.includes("image") || lower.includes("ocr")) return Image;
  if (lower.includes("knowledge") || lower.includes("search") || lower.includes("rag")) return FileSearch;
  if (lower.includes("file") || lower.includes("document") || lower.includes("read") || lower.includes("write"))
    return FileText;
  return Wrench;
}

function StatusBadge({ status }: { status: ToolStatus }) {
  if (status === "running") {
    return <Loader2 className="size-3 animate-spin text-primary shrink-0" />;
  }
  if (status === "succeeded") {
    return <Check className="size-3 text-emerald-400 shrink-0" />;
  }
  return <X className="size-3 text-rose-400 shrink-0" />;
}

export function ToolActivity({ tools }: ToolActivityProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!tools || tools.length === 0) return null;

  const hasRunning = tools.some((t) => t.status === "running");
  const hasFailed = tools.some((t) => t.status === "failed");

  return (
    <div className="my-2 max-w-full rounded-lg border border-border/70 bg-surface/40 overflow-hidden text-xs">
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-surface/70 transition-colors"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2">
          {hasRunning ? (
            <Loader2 className="size-3.5 animate-spin text-primary" />
          ) : hasFailed ? (
            <X className="size-3.5 text-rose-400" />
          ) : (
            <Check className="size-3.5 text-emerald-400" />
          )}
          <span className="font-medium text-foreground">
            Agent activity ({tools.length} step{tools.length > 1 ? "s" : ""})
          </span>
          {hasRunning && (
            <span className="rounded bg-primary/15 px-1.5 py-0.2 text-[10px] text-primary font-medium">
              In progress
            </span>
          )}
        </div>

        <span className="text-muted-foreground">
          {isOpen ? <ChevronUp className="size-3.5" /> : <ChevronDown className="size-3.5" />}
        </span>
      </button>

      {isOpen && (
        <div className="border-t border-border/50 px-3 py-2 space-y-1.5 bg-background/50">
          {tools.map((tool) => {
            const Icon = getToolIcon(tool.label);
            return (
              <div key={tool.id} className="flex items-start gap-2.5 py-1">
                <StatusBadge status={tool.status} />
                <Icon className="size-3.5 text-muted-foreground mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-foreground font-medium text-[11px] leading-tight">
                    {tool.label}
                  </div>
                  {tool.detail && (
                    <div className="mt-0.5 font-mono text-[10px] text-muted-foreground break-all">
                      {tool.detail}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
