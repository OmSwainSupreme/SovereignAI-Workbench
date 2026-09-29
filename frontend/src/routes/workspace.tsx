import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app/app-shell";
import { TaskView } from "@/features/workspace/TaskView";

export const Route = createFileRoute("/workspace")({
  head: () => ({
    meta: [
      { title: "Agent Workspace — SovereignAI Workbench" },
      {
        name: "description",
        content:
          "Run private, local AI tasks with visible tool activity, policy states, and a full audit trail.",
      },
      { property: "og:title", content: "Agent Workspace — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Run private, local AI tasks with visible tool activity and a full audit trail.",
      },
    ],
  }),
  component: WorkspacePage,
});

function WorkspacePage() {
  return (
    <AppShell>
      <TaskView taskId="new" title="New task" />
    </AppShell>
  );
}
