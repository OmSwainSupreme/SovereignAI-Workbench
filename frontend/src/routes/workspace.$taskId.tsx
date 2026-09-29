import { createFileRoute } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { ErrorState, LoadingState } from "@/components/app/states";
import { TaskView } from "@/features/workspace/TaskView";
import { errorMessage } from "@/lib/http";
import { tasksService } from "@/services/api";

const taskQuery = (taskId: string) =>
  queryOptions({
    queryKey: ["task", taskId],
    queryFn: () => tasksService.get(taskId),
  });

export const Route = createFileRoute("/workspace/$taskId")({
  head: () => ({
    meta: [
      { title: "Task — SovereignAI Workbench" },
      {
        name: "description",
        content: "Review a local AI task, its tool activity, and its outcome.",
      },
      { property: "og:title", content: "Task — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Review a local AI task, its tool activity, and its outcome.",
      },
    ],
  }),
  component: TaskPage,
});

function TaskPage() {
  const { taskId } = Route.useParams();
  const { data, isPending, error, refetch } = useQuery(taskQuery(taskId));

  if (isPending) {
    return (
      <AppShell>
        <PageContainer>
          <LoadingState rows={4} />
        </PageContainer>
      </AppShell>
    );
  }

  if (error || !data) {
    return (
      <AppShell>
        <PageContainer>
          <ErrorState
            title="This task didn't load"
            message={errorMessage(error)}
            onRetry={() => void refetch()}
          />
        </PageContainer>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <TaskView
        key={data.id}
        taskId={data.id}
        title={data.title}
        initialMessages={data.messages}
        initialToolEvents={data.toolEvents}
      />
    </AppShell>
  );
}
