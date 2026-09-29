import { createFileRoute, Link } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Library } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { errorMessage } from "@/lib/http";
import { knowledgeService } from "@/services/api";

const collectionsQuery = queryOptions({
  queryKey: ["knowledge", "collections"],
  queryFn: () => knowledgeService.collections(),
});

export const Route = createFileRoute("/knowledge/$collectionId")({
  head: () => ({
    meta: [
      { title: "Collection — SovereignAI Workbench" },
      {
        name: "description",
        content: "View a knowledge base collection stored on this machine.",
      },
      { property: "og:title", content: "Collection — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "View a knowledge base collection stored on this machine.",
      },
    ],
  }),
  component: CollectionPage,
});

function CollectionPage() {
  const { collectionId } = Route.useParams();
  const { data, isPending, error, refetch } = useQuery(collectionsQuery);
  const collection = (data ?? []).find((c) => c.id === collectionId);

  return (
    <AppShell>
      <PageContainer>
        <Link
          to="/knowledge"
          className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ArrowLeft className="size-3.5" aria-hidden />
          Knowledge base
        </Link>

        {isPending ? <LoadingState rows={3} /> : null}
        {error ? (
          <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
        ) : null}

        {!isPending && !error && !collection ? (
          <EmptyState
            icon={<Library className="size-6" />}
            title="This collection isn't available"
            description="It may have been removed, or it isn't accessible from here."
          />
        ) : null}

        {collection ? (
          <>
            <SectionHeading
              title={collection.name}
              description={`${collection.documentCount} documents · updated ${new Date(
                collection.updatedAt,
              ).toLocaleString()}`}
            />
            <EmptyState
              className="mt-6"
              icon={<Library className="size-6" />}
              title="Document listing needs the backend contract"
              description="Once the knowledge document listing API is documented, indexed documents appear here."
            />
          </>
        ) : null}
      </PageContainer>
    </AppShell>
  );
}
