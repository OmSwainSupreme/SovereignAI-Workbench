import { useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ImageIcon, ScanLine } from "lucide-react";
import { AppShell, PageContainer } from "@/components/app/app-shell";
import { EmptyState, ErrorState, SectionHeading } from "@/components/app/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage } from "@/lib/http";
import { visionService } from "@/services/api";
import type { VisionResult } from "@/services/types";

export const Route = createFileRoute("/vision")({
  head: () => ({
    meta: [
      { title: "Vision & OCR — SovereignAI Workbench" },
      {
        name: "description",
        content: "Read text and structure out of images locally, with clear, structured results.",
      },
      { property: "og:title", content: "Vision & OCR — SovereignAI Workbench" },
      {
        property: "og:description",
        content: "Read text and structure out of images locally, with clear, structured results.",
      },
    ],
  }),
  component: VisionPage,
});

function ResultPanel({ result }: { result: VisionResult }) {
  const labelled = Boolean(result.extracted || result.interpretation);

  if (!labelled) {
    return (
      <div className="rounded-lg border border-border/70 bg-surface px-5 py-4">
        <p className="text-sm leading-relaxed text-foreground">
          {result.summary ?? "No result was returned."}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {result.extracted ? (
        <div className="rounded-lg border border-border/70 bg-surface px-5 py-4">
          <h3 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {result.extracted.label}
          </h3>
          <dl className="mt-3 grid gap-2 sm:grid-cols-2">
            {result.extracted.fields.map((field) => (
              <div key={field.key} className="rounded-md bg-elevated/60 px-3 py-2">
                <dt className="text-xs text-muted-foreground">{field.key}</dt>
                <dd className="text-mono mt-0.5 text-sm text-foreground">{field.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}

      {result.interpretation ? (
        <div className="rounded-lg border border-info/30 bg-info/8 px-5 py-4">
          <h3 className="text-xs font-medium uppercase tracking-wide text-info">
            Model interpretation
          </h3>
          <p className="mt-2 text-sm leading-relaxed text-foreground">
            {result.interpretation}
          </p>
        </div>
      ) : null}
    </div>
  );
}

function VisionPage() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<VisionResult | null>(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setResult(null);
    setError(null);
    setFileName(file.name);
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return URL.createObjectURL(file);
    });
    setProcessing(true);
    try {
      setResult(await visionService.process(file));
    } catch (processError) {
      setError(errorMessage(processError));
    } finally {
      setProcessing(false);
    }
  };

  return (
    <AppShell>
      <PageContainer className="flex min-h-full max-w-none flex-col py-5 lg:py-6">
        <SectionHeading
          title="Vision & OCR"
          description="Process an image on this machine. Nothing leaves your device."
          actions={
            <>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={(event) => {
                  void handleFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              <Button onClick={() => inputRef.current?.click()} disabled={processing}>
                <ImageIcon className="size-4" aria-hidden />
                Choose image
              </Button>
            </>
          }
        />

        <div className="mt-5 grid min-h-[34rem] flex-1 gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(20rem,0.85fr)]">
          <section className="flex min-h-[22rem] min-w-0 flex-col">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Image
            </h2>
            {previewUrl ? (
              <figure className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border border-border/70 bg-surface">
                <div className="flex min-h-0 flex-1 items-center justify-center bg-elevated/20 p-3">
                <img
                  src={previewUrl}
                  alt={fileName ? `Preview of ${fileName}` : "Selected image preview"}
                  className="h-full max-h-[calc(100dvh-15rem)] w-full object-contain"
                />
                </div>
                <figcaption className="text-mono border-t border-border/60 px-3 py-2 text-xs text-muted-foreground">
                  {fileName}
                </figcaption>
              </figure>
            ) : (
              <EmptyState
                icon={<ImageIcon className="size-6" />}
                title="No image selected"
                description="Choose a scan, screenshot, or photo to process locally."
                 className="min-h-0 flex-1 py-10"
                action={
                  <Button variant="outline" onClick={() => inputRef.current?.click()}>
                    Choose image
                  </Button>
                }
              />
            )}
          </section>

          <section className="flex min-h-[22rem] min-w-0 flex-col">
            <h2 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Result
            </h2>
            <div className="min-h-0 flex-1 overflow-y-auto">
            {processing ? (
              <div
                className="h-full rounded-lg border border-border/70 bg-surface/60 p-4"
                aria-busy="true"
                aria-live="polite"
              >
                <p className="flex items-center gap-2 text-sm text-muted-foreground">
                  <ScanLine className="size-4 animate-pulse text-primary" aria-hidden />
                  Reading the image…
                </p>
                <Skeleton className="mt-4 h-3.5 w-3/4" />
                <Skeleton className="mt-2.5 h-3.5 w-1/2" />
              </div>
            ) : null}
            {error ? <ErrorState message={error} /> : null}
            {!processing && !error && result ? <ResultPanel result={result} /> : null}
            {!processing && !error && !result ? (
              <EmptyState
                icon={<ScanLine className="size-6" />}
                title="Nothing processed yet"
                description="Results appear here once an image has been read."
                 className="min-h-full py-10"
              />
            ) : null}
            </div>
          </section>
        </div>
      </PageContainer>
    </AppShell>
  );
}
