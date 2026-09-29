import React, { useState } from "react";
import { Check, Copy, Play, Loader2, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { codeService } from "@/services/api";
import type { CodeRun } from "@/services/types";
import { ExecutionResult } from "./ExecutionResult";

interface CodeBlockProps {
  language?: string;
  value: string;
}

const KEYWORDS = new Set([
  "def", "class", "return", "import", "from", "if", "else", "elif", "for", "while",
  "try", "except", "finally", "with", "as", "pass", "break", "continue", "lambda", "yield",
  "async", "await", "const", "let", "var", "function", "struct", "template", "typename",
  "public", "private", "protected", "virtual", "override", "namespace", "using",
  "int", "float", "double", "bool", "void", "auto", "new", "delete"
]);

const CONSTANTS = new Set(["true", "false", "null", "None", "self", "this"]);

function highlightLine(line: string): React.ReactNode {
  if (!line) return " ";
  // Comments
  const commentIdx = line.indexOf("#") !== -1 ? line.indexOf("#") : line.indexOf("//");
  if (commentIdx !== -1 && !line.slice(0, commentIdx).includes('"') && !line.slice(0, commentIdx).includes("'")) {
    const codePart = line.slice(0, commentIdx);
    const commentPart = line.slice(commentIdx);
    return (
      <>
        {codePart ? highlightLine(codePart) : null}
        <span className="text-zinc-500 italic">{commentPart}</span>
      </>
    );
  }

  const tokenRegex = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`|\b[a-zA-Z_]\w*\b|\d+(?:\.\d+)?|[^\s\w]+|\s+)/g;
  const parts: React.ReactNode[] = [];
  let match: RegExpExecArray | null;
  let idx = 0;

  while ((match = tokenRegex.exec(line)) !== null) {
    const text = match[0];
    if (text.startsWith('"') || text.startsWith("'") || text.startsWith("`")) {
      parts.push(<span key={idx++} className="text-emerald-300">{text}</span>);
    } else if (KEYWORDS.has(text)) {
      parts.push(<span key={idx++} className="text-purple-400 font-semibold">{text}</span>);
    } else if (CONSTANTS.has(text)) {
      parts.push(<span key={idx++} className="text-amber-300">{text}</span>);
    } else if (/^\d+(?:\.\d+)?$/.test(text)) {
      parts.push(<span key={idx++} className="text-amber-200">{text}</span>);
    } else {
      parts.push(<span key={idx++}>{text}</span>);
    }
  }

  return parts.length > 0 ? parts : line;
}

export function CodeBlock({ language = "text", value }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState<CodeRun | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const cleanCode = value.replace(/\n$/, "");
  const lines = cleanCode.split("\n");
  const isLong = lines.length > 20;
  const normalizedLang = language.toLowerCase().trim();
  const canExecute = normalizedLang === "python" || normalizedLang === "py";

  const handleCopy = () => {
    navigator.clipboard.writeText(cleanCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRun = async () => {
    if (isExecuting) return;
    setIsExecuting(true);
    try {
      const result = await codeService.execute(cleanCode, "python");
      setExecutionResult(result);
    } catch (err: any) {
      setExecutionResult({
        id: `err-${Date.now()}`,
        title: "Execution Error",
        status: "failed",
        startedAt: new Date().toISOString(),
        stderr: err?.message || "Failed to execute code in sandbox.",
        exitCode: 1,
      });
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="my-3.5 rounded-xl border border-border/80 bg-zinc-950 overflow-hidden shadow-sm text-xs">
      {/* Code Header Bar */}
      <div className="flex items-center justify-between px-3.5 py-2 border-b border-border/60 bg-surface/70 backdrop-blur-xs">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
            {language}
          </span>
          <span className="text-zinc-600 text-[10px]">·</span>
          <span className="text-[10px] text-muted-foreground font-mono">
            {lines.length} line{lines.length > 1 ? "s" : ""}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {canExecute && (
            <button
              type="button"
              onClick={handleRun}
              disabled={isExecuting}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-all cursor-pointer disabled:opacity-50"
              title="Execute in isolated Docker sandbox"
            >
              {isExecuting ? (
                <>
                  <Loader2 className="size-3 animate-spin" />
                  <span>Running...</span>
                </>
              ) : (
                <>
                  <Play className="size-3 fill-current" />
                  <span>Run Sandbox</span>
                </>
              )}
            </button>
          )}

          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-muted-foreground hover:text-foreground hover:bg-surface transition-colors cursor-pointer"
            title="Copy snippet"
          >
            {copied ? (
              <>
                <Check className="size-3 text-emerald-400" />
                <span className="text-emerald-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="size-3" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Code Content with syntax coloring & line numbers */}
      <div
        className={cn(
          "p-3.5 overflow-x-auto font-mono text-[12px] leading-relaxed text-zinc-200 selection:bg-primary/30",
          !isExpanded && isLong && "max-h-80 overflow-y-hidden relative"
        )}
      >
        <pre className="m-0 whitespace-pre">
          <code>
            {lines.map((line, i) => (
              <div key={i} className="table-row">
                <span className="table-cell select-none text-right pr-4 text-zinc-600 text-[11px] w-8">
                  {i + 1}
                </span>
                <span className="table-cell">{highlightLine(line)}</span>
              </div>
            ))}
          </code>
        </pre>
        {!isExpanded && isLong && (
          <div className="absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-zinc-950 to-transparent pointer-events-none" />
        )}
      </div>

      {/* Expand / Collapse Bar */}
      {isLong && (
        <div className="border-t border-border/40 bg-surface/30 px-3 py-1 flex justify-center">
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors cursor-pointer py-0.5"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="size-3" /> Show less
              </>
            ) : (
              <>
                <ChevronDown className="size-3" /> Show all ({lines.length} lines)
              </>
            )}
          </button>
        </div>
      )}

      {/* Sandboxed Execution Result */}
      {executionResult && (
        <div className="p-3 border-t border-border/60 bg-black/40">
          <ExecutionResult run={executionResult} onClose={() => setExecutionResult(null)} />
        </div>
      )}
    </div>
  );
}
