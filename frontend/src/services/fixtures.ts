/**
 * Sample data used ONLY while the backend API contracts are undocumented.
 *
 * These fixtures exist so the interface can be built and tested; they are
 * labelled as sample data in the UI and are deleted as each real contract
 * lands. They never simulate a real policy, authorization, or security
 * outcome — the backend is the only authority for those.
 */
import type {
  ActivityEventItem,
  CodeRun,
  KnowledgeCollection,
  Task,
  TaskSummary,
  VisionResult,
  WorkspaceEntry,
} from "./types";

const now = Date.UTC(2026, 8, 7, 12, 0, 0);
const at = (minutesAgo: number) => new Date(now - minutesAgo * 60_000).toISOString();

export const fixtureTasks: TaskSummary[] = [
  {
    id: "t-quarterly-summary",
    title: "Summarise the quarterly report",
    status: "completed",
    updatedAt: at(12),
    preview: "Produced a DOCX summary from the uploaded report.",
  },
  {
    id: "t-invoice-extract",
    title: "Extract fields from scanned invoices",
    status: "completed",
    updatedAt: at(95),
    preview: "OCR run over 4 images with structured field output.",
  },
  {
    id: "t-cleanup-script",
    title: "Run the log cleanup script",
    status: "failed",
    updatedAt: at(220),
    preview: "Sandbox run exited with a non-zero status.",
  },
  {
    id: "t-policy-blocked",
    title: "Export contact list",
    status: "denied",
    updatedAt: at(400),
    preview: "The action was not permitted.",
  },
];

export const fixtureTask: Task = {
  ...fixtureTasks[0]!,
  messages: [
    {
      id: "m1",
      role: "user",
      content: "Summarise the quarterly report and produce a DOCX I can share.",
      createdAt: at(15),
      attachments: ["q3-report.pdf"],
    },
    {
      id: "m2",
      role: "agent",
      content:
        "I read the report, pulled the headline figures, and generated a shareable summary document. Revenue grew across all three regions, with the largest movement in operations costs.",
      createdAt: at(12),
      grounded: true,
      sources: [
        {
          id: "s1",
          title: "q3-report.pdf — Financial summary",
          snippet: "Total revenue for the quarter increased against the prior period…",
          score: 0.86,
          collection: "Finance",
        },
      ],
    },
  ],
  toolEvents: [
    { id: "te1", label: "Read workspace file", status: "succeeded", detail: "q3-report.pdf" },
    { id: "te2", label: "Knowledge base lookup", status: "succeeded", detail: "3 passages" },
    { id: "te3", label: "Generate document", status: "succeeded", detail: "summary.docx" },
  ],
};

export const fixtureFiles: WorkspaceEntry[] = [
  {
    id: "f1",
    name: "q3-report.pdf",
    kind: "file",
    origin: "user_upload",
    sizeBytes: 1_482_000,
    updatedAt: at(30),
    contentType: "application/pdf",
    handle: "h-q3-report",
    downloadable: true,
  },
  {
    id: "f2",
    name: "invoice-scan-01.png",
    kind: "file",
    origin: "user_upload",
    sizeBytes: 842_000,
    updatedAt: at(96),
    contentType: "image/png",
    handle: "h-invoice-01",
    downloadable: true,
  },
  {
    id: "f3",
    name: "summary.docx",
    kind: "file",
    origin: "generated",
    sizeBytes: 38_400,
    updatedAt: at(12),
    contentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    handle: "h-summary-docx",
    downloadable: true,
  },
  {
    id: "f4",
    name: "extracted-fields.json",
    kind: "file",
    origin: "generated",
    sizeBytes: 4_200,
    updatedAt: at(94),
    contentType: "application/json",
    handle: "h-extracted",
    downloadable: true,
  },
];

export const fixtureCollections: KnowledgeCollection[] = [
  { id: "c-finance", name: "Finance", documentCount: 14, updatedAt: at(120) },
  { id: "c-policies", name: "Internal handbook", documentCount: 42, updatedAt: at(2400) },
  { id: "c-research", name: "Research notes", documentCount: 7, updatedAt: at(600) },
];

export const fixtureVision: VisionResult = {
  id: "v1",
  status: "completed",
  extracted: {
    label: "Extracted by OCR",
    fields: [
      { key: "Document type", value: "Invoice" },
      { key: "Reference", value: "INV-2291" },
      { key: "Date", value: "2026-08-19" },
      { key: "Total", value: "1,248.00" },
    ],
  },
  interpretation:
    "This looks like a standard supplier invoice with a single line item and no discount applied.",
};

export const fixtureCodeRuns: CodeRun[] = [
  {
    id: "r-cleanup",
    title: "Log cleanup script",
    status: "failed",
    startedAt: at(220),
    durationMs: 1840,
    exitCode: 1,
    stdout: "scanning archive directory...\n2 candidates found\n",
    stderr: "error: target directory is not writable\n",
  },
  {
    id: "r-parse",
    title: "Parse extracted fields",
    status: "succeeded",
    startedAt: at(94),
    durationMs: 640,
    exitCode: 0,
    stdout: "parsed 4 records\nwrote extracted-fields.json\n",
  },
  {
    id: "r-current",
    title: "Aggregate quarterly figures",
    status: "running",
    startedAt: at(1),
  },
];

export const fixtureActivity: ActivityEventItem[] = [
  { id: "a1", timestamp: at(11), action: "Document generated", status: "ok", detail: "summary.docx" },
  { id: "a2", timestamp: at(12), action: "Knowledge base queried", status: "ok", detail: "Finance" },
  {
    id: "a3",
    timestamp: at(13),
    action: "Workspace file read",
    status: "ok",
    redactedFields: ["path"],
  },
  { id: "a4", timestamp: at(95), action: "OCR processing", status: "ok", redactedFields: ["content"] },
  {
    id: "a5",
    timestamp: at(220),
    action: "Sandbox execution",
    status: "failed",
    detail: "Exited with a non-zero status",
  },
  {
    id: "a6",
    timestamp: at(400),
    action: "Export requested",
    status: "denied",
    detail: "The action was not permitted.",
  },
];
