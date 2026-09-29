import { PanelLeft, PanelLeftClose } from "lucide-react";
import { cn } from "@/lib/utils";
import { useSidebar } from "./SidebarContext";

interface ChatHeaderProps {
  onToggleSidebar?: () => void;
  isSidebarOpen?: boolean;
  selectedModel?: string | null;
  phaseLabel?: string | null;
  title?: string;
}

function formatModelName(raw?: string | null): string {
  if (!raw) return "SovereignAI";
  if (raw === "fast-path") return "Instant · Local";
  if (raw.includes("qwen3")) return "Qwen3 4B · Local";
  if (raw.includes("coder")) return "Qwen2.5-Coder 3B · Local";
  if (raw.includes("vl")) return "Qwen2.5-VL 3B · Local";
  return `${raw} · Local`;
}

export function ChatHeader({
  onToggleSidebar,
  isSidebarOpen: propIsOpen,
  selectedModel,
  phaseLabel,
  title,
}: ChatHeaderProps) {
  const sidebarCtx = useSidebar();
  const toggle = onToggleSidebar || sidebarCtx.toggleSidebar;
  const isOpen = propIsOpen !== undefined ? propIsOpen : sidebarCtx.isSidebarOpen;

  return (
    <header className="sticky top-0 z-10 flex h-14 w-full items-center justify-between border-b border-border/60 bg-background/80 px-4 backdrop-blur-md shrink-0">
      <div className="flex items-center gap-2.5 min-w-0">
        {/* Toggle button: visible on mobile always, on desktop only when sidebar is closed */}
        <button
          type="button"
          onClick={toggle}
          className={cn(
            "inline-flex size-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground hover:bg-surface hover:text-foreground transition-colors cursor-pointer",
            isOpen && "md:hidden"
          )}
          title={isOpen ? "Collapse sidebar" : "Expand sidebar (Ctrl+B)"}
        >
          {isOpen ? <PanelLeftClose className="size-4" /> : <PanelLeft className="size-4" />}
        </button>

        <div className="flex items-center gap-2 min-w-0">
          <span className="font-semibold text-sm tracking-tight text-foreground truncate max-w-[200px] sm:max-w-xs md:max-w-md">
            {title ? title : isOpen ? "New chat" : "SovereignAI"}
          </span>
          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-400 border border-emerald-500/20">
            <span className="size-1.5 rounded-full bg-emerald-400" />
            <span>Local</span>
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {selectedModel && (
          <div className="hidden sm:inline-flex items-center gap-1.5 rounded-md border border-border/60 bg-surface/60 px-2.5 py-1 text-xs font-mono text-muted-foreground">
            <span className="size-1.5 rounded-full bg-primary" />
            <span>{formatModelName(selectedModel)}</span>
          </div>
        )}

        {phaseLabel && (
          <div className="text-xs text-muted-foreground font-sans hidden md:block">
            {phaseLabel}
          </div>
        )}
      </div>
    </header>
  );
}
