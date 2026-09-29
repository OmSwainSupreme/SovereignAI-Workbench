import { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  ShieldCheck,
  Plus,
  Search,
  X,
  MoreVertical,
  Edit2,
  Trash2,
  Settings,
  User,
  Check,
  PanelLeftClose,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Conversation } from "@/hooks/useConversations";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeId?: string;
  groupedConversations: Record<"Today" | "Yesterday" | "Previous 7 Days" | "Older", Conversation[]>;
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onRename: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
  onNewChat: () => void;
}

export function Sidebar({
  isOpen,
  onClose,
  activeId,
  groupedConversations,
  searchQuery,
  onSearchChange,
  onRename,
  onDelete,
  onNewChat,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  const handleStartRename = (conv: Conversation) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
    setMenuOpenId(null);
  };

  const handleSaveRename = (id: string) => {
    if (editTitle.trim()) {
      onRename(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const hasAnyConversations = Object.values(groupedConversations).some((g) => g.length > 0);

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-xs md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar container */}
      <aside
        className={cn(
          "fixed top-0 bottom-0 left-0 z-40 flex flex-col h-full bg-sidebar border-r border-sidebar-border transition-[width,opacity] duration-250 ease-in-out md:static shrink-0 overflow-hidden",
          isOpen
            ? "w-[260px] translate-x-0 opacity-100 shadow-2xl md:shadow-none"
            : "w-0 -translate-x-full md:translate-x-0 md:w-0 border-r-0 opacity-0 pointer-events-none"
        )}
      >
        <div className="flex flex-col h-full w-[260px] min-w-[260px]">
        {/* Header */}
        <div className="flex items-center justify-between p-3.5 border-b border-sidebar-border">
          <div className="flex items-center gap-2.5">
            <div className="flex size-7 items-center justify-center rounded-lg bg-primary/15 text-primary">
              <ShieldCheck className="size-4" />
            </div>
            <span className="font-semibold text-sm tracking-tight text-sidebar-foreground">
              SovereignAI
            </span>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground transition-colors"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="size-4" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            type="button"
            onClick={() => {
              onNewChat();
              if (window.innerWidth < 768) onClose();
            }}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary text-primary-foreground py-2.5 px-3 font-medium text-xs shadow-xs hover:bg-primary/90 transition-all cursor-pointer"
          >
            <Plus className="size-4" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Search */}
        <div className="px-3 pb-2">
          <div className="relative flex items-center">
            <Search className="absolute left-2.5 size-3.5 text-muted-foreground pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search conversations..."
              className="w-full rounded-lg border border-sidebar-border bg-background/50 pl-8 pr-7 py-1.5 text-xs text-sidebar-foreground placeholder:text-muted-foreground/60 focus:outline-none focus:border-primary/50"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => onSearchChange("")}
                className="absolute right-2 text-muted-foreground hover:text-sidebar-foreground"
              >
                <X className="size-3" />
              </button>
            )}
          </div>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-4">
          {!hasAnyConversations ? (
            <div className="py-8 text-center text-xs text-muted-foreground px-4">
              {searchQuery ? "No matching conversations." : "No conversation history."}
            </div>
          ) : (
            (["Today", "Yesterday", "Previous 7 Days", "Older"] as const).map((group) => {
              const list = groupedConversations[group];
              if (!list || list.length === 0) return null;

              return (
                <div key={group} className="space-y-1">
                  <div className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                    {group}
                  </div>

                  <ul className="space-y-0.5">
                    {list.map((conv) => {
                      const isActive = activeId === conv.id;
                      const isEditing = editingId === conv.id;
                      const isMenuOpen = menuOpenId === conv.id;

                      if (isEditing) {
                        return (
                          <li key={conv.id} className="px-2 py-1">
                            <div className="flex items-center gap-1">
                              <input
                                ref={editInputRef}
                                type="text"
                                value={editTitle}
                                onChange={(e) => setEditTitle(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === "Enter") handleSaveRename(conv.id);
                                  if (e.key === "Escape") setEditingId(null);
                                }}
                                className="flex-1 rounded border border-primary bg-background px-2 py-1 text-xs text-foreground focus:outline-none"
                              />
                              <button
                                type="button"
                                onClick={() => handleSaveRename(conv.id)}
                                className="p-1 text-emerald-400 hover:text-emerald-300"
                              >
                                <Check className="size-3.5" />
                              </button>
                              <button
                                type="button"
                                onClick={() => setEditingId(null)}
                                className="p-1 text-muted-foreground hover:text-foreground"
                              >
                                <X className="size-3.5" />
                              </button>
                            </div>
                          </li>
                        );
                      }

                      return (
                        <li key={conv.id} className="relative group">
                          <Link
                            to="/workspace/$taskId"
                            params={{ taskId: conv.id }}
                            onClick={() => {
                              if (window.innerWidth < 768) onClose();
                            }}
                            className={cn(
                              "flex items-center justify-between gap-2 rounded-lg px-2.5 py-2 text-xs transition-colors pr-8",
                              isActive
                                ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                                : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                            )}
                          >
                            <span className="truncate">{conv.title}</span>
                          </Link>

                          {/* Menu trigger */}
                          <div className="absolute right-1.5 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpenId(isMenuOpen ? null : conv.id);
                              }}
                              className="p-1 rounded text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground"
                              title="More options"
                            >
                              <MoreVertical className="size-3.5" />
                            </button>
                          </div>

                          {/* Dropdown Menu */}
                          {isMenuOpen && (
                            <div
                              className="absolute right-2 top-8 z-50 w-28 rounded-lg border border-border bg-popover py-1 shadow-lg text-xs"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <button
                                type="button"
                                onClick={() => handleStartRename(conv)}
                                className="flex w-full items-center gap-2 px-3 py-1.5 text-foreground hover:bg-accent text-left"
                              >
                                <Edit2 className="size-3 text-muted-foreground" />
                                <span>Rename</span>
                              </button>
                              <button
                                type="button"
                                onClick={() => {
                                  onDelete(conv.id);
                                  setMenuOpenId(null);
                                }}
                                className="flex w-full items-center gap-2 px-3 py-1.5 text-rose-400 hover:bg-rose-500/10 text-left"
                              >
                                <Trash2 className="size-3" />
                                <span>Delete</span>
                              </button>
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-sidebar-border bg-sidebar/50 space-y-2">
          {/* Local AI status */}
          <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-surface/50 border border-sidebar-border text-[11px]">
            <span className="relative flex size-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full size-2 bg-emerald-400" />
            </span>
            <div className="flex-1 min-w-0">
              <div className="font-medium text-foreground text-xs leading-none">Local AI</div>
              <div className="text-muted-foreground text-[10px] mt-0.5">Running on-premise</div>
            </div>
            <Cpu className="size-3.5 text-muted-foreground" />
          </div>

          {/* User / Settings buttons */}
          <div className="flex items-center justify-between pt-1 text-xs text-muted-foreground">
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="flex items-center gap-1.5 p-1.5 rounded hover:bg-sidebar-accent hover:text-sidebar-foreground transition-colors"
            >
              <Settings className="size-3.5" />
              <span>Settings</span>
            </button>
            <div className="flex items-center gap-1.5 p-1.5 text-muted-foreground">
              <User className="size-3.5" />
              <span>Operator</span>
            </div>
          </div>
        </div>
        </div>
      </aside>

      {/* Simple Settings Modal */}
      {settingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-xl border border-border bg-surface p-5 shadow-xl text-xs space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <Settings className="size-4 text-primary" />
                <span>SovereignAI Settings</span>
              </h3>
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="size-4" />
              </button>
            </div>

            <div className="space-y-3 text-muted-foreground leading-relaxed">
              <div>
                <span className="font-semibold text-foreground">Runtime Engine:</span> Local Ollama & Docker
              </div>
              <div>
                <span className="font-semibold text-foreground">Backend Endpoint:</span> http://127.0.0.1:8000
              </div>
              <div>
                <span className="font-semibold text-foreground">Governance:</span> Policy Engine enforced with local loopback verification
              </div>
              <div>
                <span className="font-semibold text-foreground">Privacy:</span> 100% On-Premise. Zero external telemetry or cloud leakage.
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setSettingsOpen(false)}
                className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
