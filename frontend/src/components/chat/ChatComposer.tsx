import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square, Loader2, UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";
import { filesService } from "@/services/api";
import { AttachmentPreview, type AttachmentItem } from "./AttachmentPreview";

interface ChatComposerProps {
  onSend: (prompt: string, attachmentNames: string[]) => void;
  onCancel?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  prefillPrompt?: string;
  onClearPrefill?: () => void;
}

export function ChatComposer({
  onSend,
  onCancel,
  isStreaming = false,
  disabled = false,
  prefillPrompt = "",
  onClearPrefill,
}: ChatComposerProps) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<AttachmentItem[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync prefillPrompt if provided
  useEffect(() => {
    if (prefillPrompt) {
      setText(prefillPrompt);
      if (onClearPrefill) onClearPrefill();
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [prefillPrompt, onClearPrefill]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [text]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    if (e.clipboardData.files && e.clipboardData.files.length > 0) {
      e.preventDefault();
      handleFiles(e.clipboardData.files);
    }
  };

  const handleFiles = async (files: FileList | File[]) => {
    const fileList = Array.from(files);
    if (fileList.length === 0) return;

    setIsUploading(true);
    try {
      // Real upload to backend workspace
      await filesService.upload(fileList);

      // Create attachment items
      const newItems: AttachmentItem[] = fileList.map((f) => {
        const isImage = /\.(png|jpe?g|webp|bmp|gif|tiff)$/i.test(f.name);
        return {
          id: `${f.name}-${Date.now()}`,
          name: f.name,
          sizeBytes: f.size,
          isImage,
          previewUrl: isImage ? URL.createObjectURL(f) : undefined,
        };
      });

      setAttachments((prev) => [...prev, ...newItems]);
    } catch (err) {
      console.error("Failed to upload files to backend:", err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveAttachment = (id: string) => {
    setAttachments((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target?.previewUrl) {
        URL.revokeObjectURL(target.previewUrl);
      }
      return prev.filter((a) => a.id !== id);
    });
  };

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed && attachments.length === 0) return;
    if (isStreaming || disabled || isUploading) return;

    const attachmentNames = attachments.map((a) => a.name);
    onSend(trimmed || "Analyze the attached file(s).", attachmentNames);

    setText("");
    setAttachments([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const canSubmit = (text.trim().length > 0 || attachments.length > 0) && !isStreaming && !isUploading;

  return (
    <div
      className={cn(
        "relative rounded-2xl border border-border/80 bg-surface/90 p-2 sm:p-2.5 shadow-lg shadow-black/20 backdrop-blur-md transition-all",
        isDragging && "border-primary ring-2 ring-primary/30 bg-primary/5",
        "focus-within:border-primary/60 focus-within:ring-1 focus-within:ring-primary/40"
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-20 flex items-center justify-center rounded-2xl bg-surface/95 border-2 border-dashed border-primary">
          <div className="flex items-center gap-2 text-primary font-medium text-sm">
            <UploadCloud className="size-5" />
            <span>Drop files here to upload to Sovereign workspace</span>
          </div>
        </div>
      )}

      {/* Attachments preview */}
      {attachments.length > 0 && (
        <div className="px-2 pt-1 pb-2">
          <AttachmentPreview attachments={attachments} onRemove={handleRemoveAttachment} />
        </div>
      )}

      {/* Textarea */}
      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        onPaste={handlePaste}
        placeholder={
          isUploading
            ? "Uploading files to local workspace..."
            : "Message SovereignAI or drop files..."
        }
        disabled={disabled || isUploading}
        className="w-full resize-none bg-transparent px-2.5 py-1.5 text-sm text-foreground placeholder:text-muted-foreground/70 focus:outline-none max-h-52 leading-relaxed"
      />

      {/* Footer controls */}
      <div className="flex items-center justify-between px-1 pt-1.5 border-t border-border/40">
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || isUploading}
            className="inline-flex items-center justify-center size-8 rounded-lg text-muted-foreground hover:text-foreground hover:bg-surface transition-colors disabled:opacity-50"
            title="Attach documents or images"
          >
            {isUploading ? (
              <Loader2 className="size-4 animate-spin text-primary" />
            ) : (
              <Paperclip className="size-4" />
            )}
          </button>

          <div className="hidden sm:flex items-center gap-1.5 text-[11px] text-muted-foreground px-2 py-0.5 rounded-full bg-background/40 border border-border/50">
            <span className="size-1.5 rounded-full bg-emerald-400" />
            <span>Local AI</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isStreaming ? (
            <button
              type="button"
              onClick={onCancel}
              className="inline-flex size-8 items-center justify-center rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 transition-colors"
              title="Stop response"
            >
              <Square className="size-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className={cn(
                "inline-flex size-8 items-center justify-center rounded-lg transition-all",
                canSubmit
                  ? "bg-primary text-primary-foreground hover:bg-primary/90 shadow-xs cursor-pointer"
                  : "bg-surface/80 text-muted-foreground/40 cursor-not-allowed"
              )}
              title="Send message"
            >
              <ArrowUp className="size-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
