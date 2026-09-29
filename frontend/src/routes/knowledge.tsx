import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { Library, Search } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { SourceList } from "@/features/knowledge/SourceList";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage } from "@/lib/http";
import { knowledgeService } from "@/services/api";
import type { KnowledgeAnswer } from "@/services/types";

const collectionsQuery = queryOptions({
  queryKey: ["knowledge", "collections"],
  queryFn: () => knowledgeService.collections(),
});

export const Route = createFileRoute("/knowledge")({
  head: () => ({
    meta: [
      { title: "Knowledge Base — SovereignAI Workbench" },
      {
        name: "description",
        content: "Query your local knowledge base and see exactly which sources were used.",
      },
      { property: "og:title", content: "Knowledge Base — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Query your local knowledge base and see exactly which sources were used.",
      },
    ],
  }),
  component: KnowledgePage,
});

function KnowledgePage() {
  const { data, isPending, error, refetch } = useQuery(collectionsQuery);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<KnowledgeAnswer | null>(null);
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);

  const ask = async () => {
    const trimmed = question.trim();
    if (!trimmed) return;
    setAsking(true);
    setAskError(null);
    setAnswer(null);
    try {
      setAnswer(await knowledgeService.query(trimmed));
    } catch (queryError) {
      setAskError(errorMessage(queryError));
    } finally {
      setAsking(false);
    }
  };

  return (
    <AppShell>
      <PageContainer>
        <SectionHeading
          title="Knowledge base"
          description="Search the documents stored on this machine. Answers show the passages they came from."
        />

        <form
          className="mt-6 flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            void ask();
          }}
        >
          <div className="relative flex-1">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask your knowledge base…"
              aria-label="Ask your knowledge base"
              className="pl-9"
            />
          </div>
          <Button type="submit" disabled={asking || question.trim().length === 0}>
            {asking ? "Searching…" : "Search"}
          </Button>
        </form>

        <div className="mt-6 space-y-6">
          {asking ? (
            <div className="rounded-lg border border-border/70 bg-surface/60 p-4" aria-busy="true">
              <Skeleton className="h-3.5 w-2/3" />
              <Skeleton className="mt-3 h-3.5 w-1/2" />
              <Skeleton className="mt-3 h-3.5 w-4/5" />
            </div>
          ) : null}

          {askError ? <ErrorState message={askError} onRetry={() => void ask()} /> : null}

          {answer ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-border/70 bg-surface px-5 py-4">
                <p className="text-sm leading-relaxed text-foreground">{answer.answer}</p>
              </div>
              <SourceList sources={answer.sources} grounded={answer.grounded} />
            </div>
          ) : null}

          <section>
            <h2 className="mb-3 text-sm font-medium text-foreground">Collections</h2>
            {isPending ? <LoadingState rows={3} /> : null}
            {error ? (
              <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
            ) : null}
            {!isPending && !error && (data ?? []).length === 0 ? (
              <EmptyState
                icon={<Library className="size-6" />}
                title="No collections yet"
                description="Add documents to your workspace to build a knowledge base."
              />
            ) : null}
            {(data ?? []).length > 0 ? (
              <ul className="grid gap-2 sm:grid-cols-2">
                {(data ?? []).map((collection) => (
                  <li key={collection.id}>
                    <Link
                      to="/knowledge/$collectionId"
                      params={{ collectionId: collection.id }}
                      className="block rounded-lg border border-border/70 bg-surface/60 px-4 py-3.5 transition-colors hover:border-primary/40 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <p className="text-sm font-medium text-foreground">{collection.name}</p>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {collection.documentCount} documents · updated{" "}
                        {new Date(collection.updatedAt).toLocaleDateString()}
                      </p>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : null}
          </section>
        </div>
      </PageContainer>
    </AppShell>
  );
}
