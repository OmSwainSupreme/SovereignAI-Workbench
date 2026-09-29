import { BookOpen } from "lucide-react";
import type { KnowledgeSource } from "@/services/types";

/**
 * Retrieved sources exactly as the backend returned them. The "grounded"
 * marker appears only when the backend says knowledge was used.
 */
export function SourceList({
  sources,
  grounded,
}: {
  sources: KnowledgeSource[];
  grounded: boolean;
}) {
  if (sources.length === 0) return null;

  return (
    <div className="rounded-lg border border-border/70 bg-surface/50 p-4">
      {grounded ? (
        <p className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-primary/35 bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
          <BookOpen className="size-3" aria-hidden />
          Grounded in your knowledge base
        </p>
      ) : null}
      <ul className="space-y-3">
        {sources.map((source) => (
          <li key={source.id} className="border-l-2 border-border pl-3">
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-sm font-medium text-foreground">{source.title}</p>
              {typeof source.score === "number" ? (
                <span className="text-mono text-xs text-muted-foreground">
                  {source.score.toFixed(2)}
                </span>
              ) : null}
            </div>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              {source.snippet}
            </p>
            {source.collection ? (
              <p className="mt-1 text-xs text-muted-foreground">{source.collection}</p>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
