import { type ReactNode } from "react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "@/components/chat/Sidebar";
import { useConversations } from "@/hooks/useConversations";
import { SidebarProvider, useSidebar } from "@/components/chat/SidebarContext";

function AppShellInner({ children }: { children: ReactNode }) {
  const { isSidebarOpen, closeSidebar } = useSidebar();
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  // Extract activeId if on /workspace/$taskId
  const pathParts = pathname.split("/").filter(Boolean);
  const activeId = pathParts.length >= 2 && pathParts[0] === "workspace" ? pathParts[1] : undefined;

  const {
    grouped,
    searchQuery,
    setSearchQuery,
    renameConversation,
    deleteConversation,
  } = useConversations(activeId);

  const handleNewChat = () => {
    navigate({ to: "/workspace" });
  };

  return (
    <div className="flex h-dvh min-h-0 w-full overflow-hidden bg-background">
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={closeSidebar}
        activeId={activeId}
        groupedConversations={grouped}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onRename={renameConversation}
        onDelete={(id) => {
          deleteConversation(id);
          if (activeId === id) {
            navigate({ to: "/workspace" });
          }
        }}
        onNewChat={handleNewChat}
      />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {children}
        </div>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <SidebarProvider>
      <AppShellInner>{children}</AppShellInner>
    </SidebarProvider>
  );
}

export function PageContainer({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={className || "mx-auto w-full max-w-4xl px-4 py-6"}>
      {children}
    </div>
  );
}
