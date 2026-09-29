import { useState } from "react";
import { Download, ExternalLink, FileText, Loader2, Check } from "lucide-react";
import { filesService } from "@/services/api";

interface ArtifactCardProps {
  filename: string;
  sizeBytes?: number;
  handle?: string;
}

export function ArtifactCard({ filename, sizeBytes, handle }: ArtifactCardProps) {
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);

  const fileHandle = handle || filename;

  const handleDownload = async () => {
    if (downloading) return;
    setDownloading(true);
    try {
      await filesService.download(fileHandle);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 2000);
    } catch (err) {
      console.error("Download failed:", err);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="my-2.5 max-w-sm rounded-xl border border-border/80 bg-surface/70 p-3.5 shadow-sm text-xs">
      <div className="flex items-start gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
          <FileText className="size-5" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="font-semibold text-foreground truncate text-sm">{filename}</div>
          <div className="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
            <span>Generated locally</span>
            {sizeBytes ? (
              <>
                <span>·</span>
                <span>{(sizeBytes / 1024).toFixed(1)} KB</span>
              </>
            ) : null}
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center gap-2 pt-2 border-t border-border/50">
        <button
          type="button"
          onClick={handleDownload}
          disabled={downloading}
          className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-md bg-primary/10 hover:bg-primary/20 text-primary py-1.5 px-3 font-medium transition-colors disabled:opacity-50"
        >
          {downloading ? (
            <>
              <Loader2 className="size-3.5 animate-spin" />
              <span>Downloading...</span>
            </>
          ) : downloaded ? (
            <>
              <Check className="size-3.5 text-emerald-400" />
              <span>Downloaded</span>
            </>
          ) : (
            <>
              <Download className="size-3.5" />
              <span>Download</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
