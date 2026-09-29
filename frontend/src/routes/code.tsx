import { createFileRoute, Link } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { Terminal } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { RunStatusPill } from "@/components/app/status";
import { errorMessage } from "@/lib/http";
import { codeService } from "@/services/api";

const runsQuery = queryOptions({ queryKey: ["code", "runs"], queryFn: () => codeService.list() });

export const Route = createFileRoute("/code")({
  head: () => ({
    meta: [
      { title: "Code Runs — SovereignAI Workbench" },
      {
        name: "description",
        content: "Track isolated code executions, their status, and their output.",
      },
      { property: "og:title", content: "Code Runs — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Track isolated code executions, their status, and their output.",
      },
    ],
  }),
  component: CodePage,
});

function CodePage() {
  const { data, isPending, error, refetch } = useQuery(runsQuery);
  const runs = data ?? [];

  return (
    <AppShell>
      <PageContainer>
        <SectionHeading
          title="Code runs"
          description="Code the agent runs is executed in isolation by the backend. You see status and output only."
        />

        <div className="mt-6">
          {isPending ? <LoadingState rows={3} /> : null}
          {error ? (
            <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
          ) : null}
          {!isPending && !error && runs.length === 0 ? (
            <EmptyState
              icon={<Terminal className="size-6" />}
              title="No code has been run yet"
              description="When a task runs code, the run and its output appear here."
            />
          ) : null}
          {runs.length > 0 ? (
            <ul className="space-y-2">
              {runs.map((run) => (
                <li key={run.id}>
                  <Link
                    to="/code/$runId"
                    params={{ runId: run.id }}
                    className="flex items-center justify-between gap-4 rounded-lg border border-border/70 bg-surface/60 px-4 py-3.5 transition-colors hover:border-primary/40 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-foreground">{run.title}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        Started {new Date(run.startedAt).toLocaleString()}
                        {run.durationMs ? ` · ${(run.durationMs / 1000).toFixed(2)}s` : ""}
                      </p>
                    </div>
                    <RunStatusPill status={run.status} />
                  </Link>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </PageContainer>
    </AppShell>
  );
}
