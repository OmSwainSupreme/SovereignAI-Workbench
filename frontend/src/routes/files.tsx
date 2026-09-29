import { useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { queryOptions, useQuery } from "@tanstack/react-query";
import { Download, FileCog, FileUp, FolderClosed } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, LoadingState, SectionHeading } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/http";
import { filesService } from "@/services/api";
import type { WorkspaceEntry } from "@/services/types";

const filesQuery = queryOptions({ queryKey: ["files"], queryFn: () => filesService.list() });

export const Route = createFileRoute("/files")({
  head: () => ({
    meta: [
      { title: "Workspace Files — SovereignAI Workbench" },
      {
        name: "description",
        content:
          "Browse your workspace files, upload new documents, and download generated artifacts.",
      },
      { property: "og:title", content: "Workspace Files — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Browse workspace files, upload documents, and download generated artifacts.",
      },
    ],
  }),
  component: FilesPage,
});

function formatSize(bytes?: number) {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

function FileRow({
  entry,
  onDownload,
  busy,
}: {
  entry: WorkspaceEntry;
  onDownload: (entry: WorkspaceEntry) => void;
  busy: boolean;
}) {
  return (
    <li className="flex items-center justify-between gap-4 rounded-lg border border-border/70 bg-surface/60 px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        {entry.origin === "generated" ? (
          <FileCog className="size-4 shrink-0 text-primary" aria-hidden />
        ) : (
          <FileUp className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        )}
        <div className="min-w-0">
          <p className="text-mono truncate text-sm text-foreground">{entry.name}</p>
          <p className="text-xs text-muted-foreground">
            {formatSize(entry.sizeBytes)} · {new Date(entry.updatedAt).toLocaleString()}
          </p>
        </div>
      </div>
      {entry.downloadable ? (
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => onDownload(entry)}
          aria-label={`Download ${entry.name}`}
        >
          <Download className="size-3.5" aria-hidden />
          Download
        </Button>
      ) : null}
    </li>
  );
}

function FilesPage() {
  const { data, isPending, error, refetch } = useQuery(filesQuery);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);

  const entries = data ?? [];
  const uploads = entries.filter((e) => e.origin === "user_upload");
  const generated = entries.filter((e) => e.origin === "generated");
  const unclassified = entries.filter((e) => e.origin === "unknown");

  const handleUpload = async (files: File[]) => {
    if (files.length === 0) return;
    setBusy(true);
    setActionError(null);
    try {
      await filesService.upload(files);
      await refetch();
    } catch (uploadError) {
      setActionError(errorMessage(uploadError));
    } finally {
      setBusy(false);
    }
  };

  const handleDownload = async (entry: WorkspaceEntry) => {
    setBusy(true);
    setActionError(null);
    try {
      await filesService.download(entry.handle);
    } catch (downloadError) {
      setActionError(errorMessage(downloadError));
    } finally {
      setBusy(false);
    }
  };

  return (
    <AppShell>
      <PageContainer>
        <SectionHeading
          title="Workspace files"
          description="Files you upload and files the agent generates. Access is checked by the backend for every action."
          actions={
            <>
              <input
                ref={uploadRef}
                type="file"
                multiple
                className="sr-only"
                onChange={(event) => {
                  void handleUpload(Array.from(event.target.files ?? []));
                  event.target.value = "";
                }}
              />
              <Button onClick={() => uploadRef.current?.click()} disabled={busy}>
                <FileUp className="size-4" aria-hidden />
                Upload files
              </Button>
            </>
          }
        />

        {actionError ? <ErrorState className="mt-6" message={actionError} /> : null}

        <div className="mt-6 space-y-8">
          {isPending ? <LoadingState rows={4} /> : null}
          {error ? (
            <ErrorState message={errorMessage(error)} onRetry={() => void refetch()} />
          ) : null}

          {!isPending && !error && entries.length === 0 ? (
            <EmptyState
              icon={<FolderClosed className="size-6" />}
              title="Your workspace is empty"
              description="Upload a document to get started. The backend validates and stores everything."
              action={
                <Button onClick={() => uploadRef.current?.click()}>Upload files</Button>
              }
            />
          ) : null}

          {uploads.length > 0 ? (
            <section>
              <h2 className="mb-3 text-sm font-medium text-foreground">Your uploads</h2>
              <ul className="space-y-2">
                {uploads.map((entry) => (
                  <FileRow
                    key={entry.id}
                    entry={entry}
                    onDownload={(e) => void handleDownload(e)}
                    busy={busy}
                  />
                ))}
              </ul>
            </section>
          ) : null}

          {generated.length > 0 ? (
            <section>
              <h2 className="mb-3 text-sm font-medium text-foreground">Generated artifacts</h2>
              <ul className="space-y-2">
                {generated.map((entry) => (
                  <FileRow
                    key={entry.id}
                    entry={entry}
                    onDownload={(e) => void handleDownload(e)}
                    busy={busy}
                  />
                ))}
              </ul>
            </section>
          ) : null}

          {unclassified.length > 0 ? (
            <section>
              <h2 className="mb-3 text-sm font-medium text-foreground">Files</h2>
              <ul className="space-y-2">
                {unclassified.map((entry) => (
                  <FileRow
                    key={entry.id}
                    entry={entry}
                    onDownload={(e) => void handleDownload(e)}
                    busy={busy}
                  />
                ))}
              </ul>
            </section>
          ) : null}
        </div>
      </PageContainer>
    </AppShell>
  );
}
