import { FileText, ImageIcon, X } from "lucide-react";
import { cn } from "@/lib/utils";

export interface AttachmentItem {
  id: string;
  name: string;
  sizeBytes?: number;
  isImage: boolean;
  previewUrl?: string; // object URL for local image preview
}

interface AttachmentPreviewProps {
  attachments: AttachmentItem[];
  onRemove?: (id: string) => void;
  readOnly?: boolean;
}

export function AttachmentPreview({ attachments, onRemove, readOnly = false }: AttachmentPreviewProps) {
  if (!attachments || attachments.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2.5 my-2">
      {attachments.map((item) => {
        if (item.isImage) {
          return (
            <div
              key={item.id}
              className="relative group rounded-xl border border-border/80 bg-surface/80 overflow-hidden shadow-sm"
            >
              <div className="size-20 bg-black/40 flex items-center justify-center overflow-hidden">
                {item.previewUrl ? (
                  <img
                    src={item.previewUrl}
                    alt={item.name}
                    className="size-full object-cover"
                  />
                ) : (
                  <ImageIcon className="size-8 text-muted-foreground" />
                )}
              </div>
              <div className="px-2 py-1 bg-surface/90 text-[10px] font-mono text-muted-foreground truncate max-w-[5rem]">
                {item.name}
              </div>

              {!readOnly && onRemove && (
                <button
                  type="button"
                  onClick={() => onRemove(item.id)}
                  className="absolute top-1 right-1 size-5 rounded-full bg-black/70 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black"
                  title="Remove image"
                >
                  <X className="size-3" />
                </button>
              )}
            </div>
          );
        }

        // Clean file chip/card for document/text
        return (
          <div
            key={item.id}
            className="relative group flex items-center gap-2.5 rounded-xl border border-border/80 bg-surface/90 px-3 py-2 text-xs shadow-sm max-w-xs"
          >
            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
              <FileText className="size-4" />
            </div>

            <div className="flex-1 min-w-0 pr-2">
              <div className="font-medium text-foreground truncate text-xs">{item.name}</div>
              {item.sizeBytes !== undefined && (
                <div className="text-[10px] font-mono text-muted-foreground">
                  {(item.sizeBytes / 1024).toFixed(1)} KB
                </div>
              )}
            </div>

            {!readOnly && onRemove && (
              <button
                type="button"
                onClick={() => onRemove(item.id)}
                className="size-5 rounded-full text-muted-foreground hover:text-foreground hover:bg-surface flex items-center justify-center transition-colors"
                title="Remove attachment"
              >
                <X className="size-3.5" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
