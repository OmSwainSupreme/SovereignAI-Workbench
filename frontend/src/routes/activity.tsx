import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { Activity as ActivityIcon, EyeOff } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { ActivityStatusPill } from "@/components/app/status";
import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/http";
import { activityService } from "@/services/api";
import type { ActivityStatus } from "@/services/types";

const activityQuery = queryOptions({
  queryKey: ["activity"],
  queryFn: () => activityService.list(),
});

export const Route = createFileRoute("/activity")({
  head: () => ({
    meta: [
      { title: "Activity — SovereignAI Workbench" },
      {
        name: "description",
        content: "A readable, privacy-preserving timeline of everything the agent has done.",
      },
      { property: "og:title", content: "Activity — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "A readable, privacy-preserving timeline of everything the agent has done.",
      },
    ],
  }),
  component: ActivityPage,
});

const FILTERS: { value: ActivityStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "ok", label: "Completed" },
  { value: "denied", label: "Not permitted" },
  { value: "failed", label: "Failed" },
];

function ActivityPage() {
  const { data, isPending, error, refetch } = useQuery(activityQuery);
  const [status, setStatus] = useState<ActivityStatus | "all">("all");

  const events = (data ?? []).filter((event) => status === "all" || event.status === status);

  const groups = events.reduce<Record<string, typeof events>>((acc, event) => {
    const day = new Date(event.timestamp).toLocaleDateString(undefined, {
      weekday: "long",
      day: "numeric",
      month: "long",
    });
    (acc[day] ??= []).push(event);
    return acc;
  }, {});

  return (
    <AppShell>
      <PageContainer>
        <SectionHeading
          title="Activity"
          description="Everything the agent did, in plain language. Sensitive details are withheld by the backend and never requested here."
        />

        <div className="mt-6 flex gap-1.5" role="group" aria-label="Filter activity">
          {FILTERS.map((filter) => (
            <Button
              key={filter.value}
              size="sm"
              variant={status === filter.value ? "secondary" : "ghost"}
              aria-pressed={status === filter.value}
              onClick={() => setStatus(filter.value)}
            >
              {filter.label}
            </Button>
          ))}
        </div>

        <div className="mt-6 space-y-8">
          {isPending ? <LoadingState rows={4} /> : null}
          {error ? (
            <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
          ) : null}
          {!isPending && !error && events.length === 0 ? (
            <EmptyState
              icon={<ActivityIcon className="size-6" />}
              title="No activity to show"
              description="Actions appear here as soon as the agent starts working."
            />
          ) : null}

          {Object.entries(groups).map(([day, dayEvents]) => (
            <section key={day}>
              <h2 className="mb-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {day}
              </h2>
              <ol className="space-y-2">
                {dayEvents.map((event) => (
                  <li
                    key={event.id}
                    className="flex items-start justify-between gap-4 rounded-lg border border-border/70 bg-surface/60 px-4 py-3"
                  >
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-foreground">{event.action}</p>
                      {event.detail ? (
                        <p className="mt-0.5 text-sm text-muted-foreground">{event.detail}</p>
                      ) : null}
                      {event.redactedFields?.length ? (
                        <p className="mt-1.5 inline-flex items-center gap-1.5 rounded border border-border bg-muted/60 px-2 py-0.5 text-xs text-muted-foreground">
                          <EyeOff className="size-3" aria-hidden />
                          Some details are withheld for privacy
                        </p>
                      ) : null}
                    </div>
                    <div className="flex shrink-0 items-center gap-3">
                      <time
                        dateTime={event.timestamp}
                        className="text-mono text-xs text-muted-foreground"
                      >
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </time>
                      <ActivityStatusPill status={event.status} />
                    </div>
                  </li>
                ))}
              </ol>
            </section>
          ))}
        </div>
      </PageContainer>
    </AppShell>
  );
}
