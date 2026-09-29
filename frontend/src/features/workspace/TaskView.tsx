import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  FileText,
  ImageIcon,
  Loader2,
  ArrowDown,
  Copy,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatMessage, TaskStatus, ToolEvent } from "@/services/types";
import { useTaskStream } from "./useTaskStream";
import { useConversations } from "@/hooks/useConversations";
import { useSidebar } from "@/components/chat/SidebarContext";
import { ChatHeader } from "@/components/chat/ChatHeader";
import { ChatComposer } from "@/components/chat/ChatComposer";
import { EmptyState } from "@/components/chat/EmptyState";
import { ThinkingBlock } from "@/components/chat/ThinkingBlock";
import { ToolActivity } from "@/components/chat/ToolActivity";
import { MarkdownRenderer } from "@/components/chat/MarkdownRenderer";
import { ArtifactCard } from "@/components/chat/ArtifactCard";
import { AttachmentPreview, type AttachmentItem } from "@/components/chat/AttachmentPreview";
import { PolicyNotice } from "@/components/app/policy-notice";
import { SourceList } from "@/features/knowledge/SourceList";
import { API_BASE_URL } from "@/lib/config";

interface TaskViewProps {
  taskId: string;
  title?: string;
  initialMessages?: ChatMessage[];
  initialToolEvents?: ToolEvent[];
}

function CopyMessageButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/70 hover:text-foreground transition-colors cursor-pointer py-0.5 px-2 rounded-md hover:bg-surface border border-transparent hover:border-border/60"
      title="Copy response"
    >
      {copied ? (
        <>
          <Check className="size-3 text-emerald-400" />
          <span className="text-emerald-400 text-[10px]">Copied</span>
        </>
      ) : (
        <>
          <Copy className="size-3" />
          <span className="text-[10px]">Copy</span>
        </>
      )}
    </button>
  );
}

export function TaskView({
  taskId,
  title = "New chat",
  initialMessages,
  initialToolEvents,
}: TaskViewProps) {
  const stream = useTaskStream({
    taskId,
    initialMessages,
    initialToolEvents,
  });

  const { addConversation } = useConversations();
  const [prefillPrompt, setPrefillPrompt] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottomRef = useRef(true);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  // Sync new task ID into URL and conversation list
  useEffect(() => {
    if (stream.currentTaskId && taskId === "new") {
      const firstUserMsg = stream.messages.find((m) => m.role === "user");
      const generatedTitle = firstUserMsg
        ? firstUserMsg.content.slice(0, 45).trim() + (firstUserMsg.content.length > 45 ? "..." : "")
        : "New conversation";

      addConversation({
        id: stream.currentTaskId,
        title: generatedTitle,
        updatedAt: new Date().toISOString(),
        status: stream.status,
      });

      // Update URL silently to match the real task ID
      window.history.replaceState(null, "", `/workspace/${stream.currentTaskId}`);
    }
  }, [stream.currentTaskId, taskId, stream.messages, stream.status, addConversation]);

  // Handle scroll detection for auto-scroll locking
  const handleScroll = () => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceToBottom < 90;
    isNearBottomRef.current = nearBottom;
    setShowScrollBottom(!nearBottom && stream.messages.length > 1);
  };

  // Auto-scroll only when user is near bottom
  useEffect(() => {
    if (isNearBottomRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [stream.messages, stream.isStreaming]);

  const scrollToBottom = () => {
    isNearBottomRef.current = true;
    setShowScrollBottom(false);
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSend = (prompt: string, attachmentNames: string[]) => {
    isNearBottomRef.current = true;
    void stream.run(prompt, attachmentNames);
  };

  const hasMessages = stream.messages.length > 0;
  const isFailed = stream.status === "failed" || Boolean(stream.error);

  return (
    <div className="flex h-full w-full flex-col bg-background overflow-hidden">
      {/* Top Header */}
      <ChatHeader
        title={title && title !== "New task" ? title : undefined}
        selectedModel={stream.model}
        phaseLabel={stream.phaseLabel}
      />

      {/* Main Conversation Scroll Area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 py-6 sm:px-6 md:py-8"
      >
        <div className="mx-auto max-w-3xl lg:max-w-4xl space-y-6">
          {!hasMessages && (
            <EmptyState onSelectPrompt={(prompt) => setPrefillPrompt(prompt)} />
          )}

          {/* Messages */}
          {stream.messages.map((message) => {
            const isUser = message.role === "user";

            if (isUser) {
              const userAttachments: AttachmentItem[] = (message.attachments || []).map((name) => {
                const isImage = /\.(png|jpe?g|webp|bmp|gif|tiff)$/i.test(name);
                return {
                  id: name,
                  name,
                  isImage,
                  previewUrl: isImage ? `${API_BASE_URL}/api/v1/files/${encodeURIComponent(name)}` : undefined,
                };
              });

              return (
                <div key={message.id} className="flex justify-end my-4">
                  <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl bg-surface/90 border border-border/70 px-4 py-3 text-sm text-foreground shadow-xs">
                    {userAttachments.length > 0 && (
                      <AttachmentPreview attachments={userAttachments} readOnly={true} />
                    )}
                    <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
                  </div>
                </div>
              );
            }

            // Assistant Message
            const hasThinking = Boolean(message.thinking && message.thinking.trim().length > 0);
            const isThinkingActive = Boolean(message.pending && !message.content);
            const hasContent = Boolean(message.content && message.content.trim().length > 0);

            // Detect generated artifacts
            const artifactMatches = message.content?.match(/[\w-]+\.(docx|pdf|txt|csv|xlsx)/gi) || [];
            const uniqueArtifacts = Array.from(new Set(artifactMatches)).filter(
              (a) => !message.attachments?.includes(a)
            );

            return (
              <div key={message.id} className="flex items-start gap-3.5 my-6">
                {/* SovereignAI Avatar */}
                <div className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-b from-primary/20 to-primary/5 border border-primary/30 text-primary shadow-xs mt-0.5">
                  <ShieldCheck className="size-4.5" />
                </div>

                <div className="flex-1 min-w-0 space-y-2">
                  {/* Thinking Block */}
                  {hasThinking && (
                    <ThinkingBlock
                      thinking={message.thinking || ""}
                      isStreaming={Boolean(message.pending)}
                      hasContent={hasContent}
                    />
                  )}

                  {/* Inline Tool Execution Activity */}
                  {stream.toolEvents && stream.toolEvents.length > 0 && (
                    <ToolActivity tools={stream.toolEvents} />
                  )}

                  {/* Assistant Text / Markdown */}
                  {hasContent ? (
                    <div className="text-foreground text-sm leading-relaxed">
                      <MarkdownRenderer content={message.content} />
                      {message.pending && (
                        <span className="ml-1 inline-block h-4 w-1.5 animate-pulse bg-primary align-text-bottom" />
                      )}
                      <div className="flex items-center gap-2 pt-2">
                        <CopyMessageButton text={message.content} />
                      </div>
                    </div>
                  ) : isThinkingActive ? (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground italic py-1">
                      <Loader2 className="size-3.5 animate-spin text-primary" />
                      <span>Thinking through the request...</span>
                    </div>
                  ) : null}

                  {/* Inline Artifact Cards (e.g. Generated Reports) */}
                  {uniqueArtifacts.length > 0 && (
                    <div className="pt-2">
                      {uniqueArtifacts.map((art) => (
                        <ArtifactCard key={art} filename={art} />
                      ))}
                    </div>
                  )}

                  {/* Grounding / Knowledge Sources */}
                  {message.sources && message.sources.length > 0 && (
                    <SourceList sources={message.sources} grounded={message.grounded ?? false} />
                  )}
                </div>
              </div>
            );
          })}

          {/* Active Tool Activity (while streaming before message finishes) */}
          {stream.isStreaming &&
            stream.toolEvents.length > 0 &&
            !stream.messages.some((m) => m.role === "agent") && (
              <div className="flex items-start gap-3.5 my-4">
                <div className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary/10 border border-primary/20 text-primary">
                  <Loader2 className="size-4 animate-spin" />
                </div>
                <div className="flex-1 min-w-0">
                  <ToolActivity tools={stream.toolEvents} />
                </div>
              </div>
            )}

          {/* Policy Notice if Denied or Approval Required */}
          {stream.policy && (
            <div className="my-3 max-w-xl">
              <PolicyNotice notice={stream.policy} />
            </div>
          )}

          {/* Critical Failure & Error Handling */}
          {isFailed && (
            <div className="my-4 rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-xs shadow-xs">
              <div className="flex items-start gap-3">
                <AlertTriangle className="size-5 text-rose-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <div className="font-semibold text-rose-300 text-sm">Something went wrong</div>
                  <div className="mt-1 text-muted-foreground leading-relaxed">
                    SovereignAI couldn't complete this request.
                  </div>
                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => stream.retry()}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 px-3 py-1.5 font-medium transition-colors cursor-pointer"
                    >
                      <RotateCcw className="size-3.5" />
                      <span>Retry</span>
                    </button>
                  </div>
                  {stream.error && (
                    <details className="mt-3 text-muted-foreground/80 font-mono text-[11px]">
                      <summary className="cursor-pointer text-xs font-sans text-muted-foreground hover:text-foreground">
                        Technical details
                      </summary>
                      <div className="mt-1.5 p-2.5 rounded-lg bg-black/40 border border-border/50 break-all whitespace-pre-wrap">
                        {stream.error}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Modern Chat Composer */}
      <div className="w-full px-4 pb-4 sm:px-6 sm:pb-6 relative">
        <div className="mx-auto max-w-3xl lg:max-w-4xl relative">
          {/* Scroll to bottom button */}
          {showScrollBottom && (
            <div className="absolute -top-10 left-1/2 -translate-x-1/2 z-20">
              <button
                type="button"
                onClick={scrollToBottom}
                className="inline-flex items-center gap-1.5 rounded-full border border-border/80 bg-surface/90 px-3 py-1.5 text-xs text-foreground shadow-md hover:bg-surface hover:border-primary/40 transition-all backdrop-blur-xs cursor-pointer"
              >
                <ArrowDown className="size-3.5 text-primary" />
                <span>Scroll to bottom</span>
              </button>
            </div>
          )}


          <ChatComposer
            onSend={handleSend}
            onCancel={stream.cancel}
            isStreaming={stream.isStreaming}
            disabled={stream.status === "running"}
            prefillPrompt={prefillPrompt}
            onClearPrefill={() => setPrefillPrompt("")}
          />
        </div>
      </div>
    </div>
  );
}
