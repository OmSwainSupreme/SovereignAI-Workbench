import { createFileRoute, Link } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { RunStatusPill } from "@/components/app/status";
import { PolicyNotice } from "@/components/app/policy-notice";
import { errorMessage } from "@/lib/http";
import { codeService } from "@/services/api";

const runQuery = (runId: string) =>
  queryOptions({ queryKey: ["code", "run", runId], queryFn: () => codeService.get(runId) });

export const Route = createFileRoute("/code/$runId")({
  head: () => ({
    meta: [
      { title: "Code Run — SovereignAI Workbench" },
      { name: "description", content: "Status and output for a single isolated code run." },
      { property: "og:title", content: "Code Run — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Status and output for a single isolated code run.",
      },
    ],
  }),
  component: RunPage,
});

function OutputBlock({ title, content, tone }: { title: string; content: string; tone: "out" | "err" }) {
  return (
    <section>
      <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {title}
      </h2>
      <pre
        className={`text-mono max-h-80 overflow-auto rounded-lg border px-4 py-3 text-xs leading-relaxed ${
          tone === "err"
            ? "border-destructive/35 bg-destructive/8 text-destructive"
            : "border-border/70 bg-elevated/60 text-foreground"
        }`}
      >
        {content}
      </pre>
    </section>
  );
}

function RunPage() {
  const { runId } = Route.useParams();
  const { data, isPending, error, refetch } = useQuery(runQuery(runId));

  return (
    <AppShell>
      <PageContainer>
        <Link
          to="/code"
          className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ArrowLeft className="size-3.5" aria-hidden />
          Code runs
        </Link>

        {isPending ? <LoadingState rows={3} /> : null}
        {error ? (
          <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
        ) : null}

        {data ? (
          <div className="space-y-6">
            <SectionHeading
              title={data.title}
              description={`Started ${new Date(data.startedAt).toLocaleString()}${
                typeof data.exitCode === "number" ? ` · exit status ${data.exitCode}` : ""
              }`}
              actions={<RunStatusPill status={data.status} />}
            />

            {data.policy ? <PolicyNotice notice={data.policy} /> : null}

            {data.status === "running" ? (
              <p className="text-sm text-muted-foreground" aria-live="polite">
                This run is still in progress. Output appears when it finishes.
              </p>
            ) : null}

            {data.stdout ? <OutputBlock title="Output" content={data.stdout} tone="out" /> : null}
            {data.stderr ? <OutputBlock title="Errors" content={data.stderr} tone="err" /> : null}

            {!data.stdout && !data.stderr && data.status !== "running" ? (
              <p className="text-sm text-muted-foreground">This run produced no output.</p>
            ) : null}
          </div>
        ) : null}
      </PageContainer>
    </AppShell>
  );
}
