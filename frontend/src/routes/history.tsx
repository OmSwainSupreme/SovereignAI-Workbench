import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { MessagesSquare, Search } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { TaskStatusPill } from "@/components/app/status";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { errorMessage } from "@/lib/http";
import { tasksService } from "@/services/api";
import type { TaskStatus } from "@/services/types";

const tasksQuery = queryOptions({ queryKey: ["tasks"], queryFn: () => tasksService.list() });

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Task History — SovereignAI Workbench" },
      {
        name: "description",
        content: "Browse and search previous local AI tasks and their outcomes.",
      },
      { property: "og:title", content: "Task History — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Browse and search previous local AI tasks and their outcomes.",
      },
    ],
  }),
  component: HistoryPage,
});

const FILTERS: { value: TaskStatus | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "completed", label: "Completed" },
  { value: "failed", label: "Failed" },
  { value: "denied", label: "Not permitted" },
];

function HistoryPage() {
  const { data, isPending, error, refetch } = useQuery(tasksQuery);
  const [term, setTerm] = useState("");
  const [status, setStatus] = useState<TaskStatus | "all">("all");

  const tasks = (data ?? []).filter(
    (task) =>
      (status === "all" || task.status === status) &&
      task.title.toLowerCase().includes(term.trim().toLowerCase()),
  );

  return (
    <AppShell>
      <PageContainer>
        <SectionHeading
          title="Task history"
          description="Every task you have run on this machine, with its final state."
        />

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <div className="relative min-w-64 flex-1">
            <Search
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={term}
              onChange={(event) => setTerm(event.target.value)}
              placeholder="Search tasks"
              aria-label="Search tasks"
              className="pl-9"
            />
          </div>
          <div className="flex gap-1.5" role="group" aria-label="Filter by status">
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
        </div>

        <div className="mt-6">
          {isPending ? <LoadingState rows={4} /> : null}
          {error ? (
            <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
          ) : null}
          {!isPending && !error && tasks.length === 0 ? (
            <EmptyState
              icon={<MessagesSquare className="size-6" />}
              title="No tasks match"
              description="Try a different search term or clear the status filter."
            />
          ) : null}
          {tasks.length > 0 ? (
            <ul className="space-y-2">
              {tasks.map((task) => (
                <li key={task.id}>
                  <Link
                    to="/workspace/$taskId"
                    params={{ taskId: task.id }}
                    className="flex items-center justify-between gap-4 rounded-lg border border-border/70 bg-surface/60 px-4 py-3.5 transition-colors hover:border-primary/40 hover:bg-surface focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-foreground">
                        {task.title}
                      </p>
                      {task.preview ? (
                        <p className="mt-0.5 truncate text-sm text-muted-foreground">
                          {task.preview}
                        </p>
                      ) : null}
                    </div>
                    <div className="flex shrink-0 items-center gap-3">
                      <span className="text-mono text-xs text-muted-foreground">
                        {new Date(task.updatedAt).toLocaleString()}
                      </span>
                      <TaskStatusPill status={task.status} />
                    </div>
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
