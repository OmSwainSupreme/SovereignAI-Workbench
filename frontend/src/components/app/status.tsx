import {
  CheckCircle2,
  CircleDashed,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  ShieldQuestion,
  XCircle,
} from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type {
  ActivityStatus,
  CodeRunStatus,
  PolicyDecision,
  TaskStatus,
  ToolStatus,
} from "@/services/types";

type Tone = "neutral" | "running" | "success" | "danger" | "warning";

const toneClass: Record<Tone, string> = {
  neutral: "border-border bg-muted/60 text-muted-foreground",
  running: "border-info/40 bg-info/12 text-info",
  success: "border-success/40 bg-success/12 text-success",
  danger: "border-destructive/45 bg-destructive/12 text-destructive",
  warning: "border-warning/40 bg-warning/12 text-warning",
};

export function StatusPill({
  tone,
  icon,
  children,
  className,
}: {
  tone: Tone;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        toneClass[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

const taskMeta: Record<TaskStatus, { label: string; tone: Tone; icon: ReactNode }> = {
  idle: { label: "Ready", tone: "neutral", icon: <CircleDashed className="size-3" /> },
  queued: { label: "Queued", tone: "neutral", icon: <CircleDashed className="size-3" /> },
  running: {
    label: "Running",
    tone: "running",
    icon: <Loader2 className="size-3 animate-spin" />,
  },
  completed: { label: "Completed", tone: "success", icon: <CheckCircle2 className="size-3" /> },
  failed: { label: "Failed", tone: "danger", icon: <XCircle className="size-3" /> },
  cancelled: { label: "Cancelled", tone: "neutral", icon: <XCircle className="size-3" /> },
  denied: { label: "Not permitted", tone: "danger", icon: <ShieldAlert className="size-3" /> },
  awaiting_approval: {
    label: "Needs approval",
    tone: "warning",
    icon: <ShieldQuestion className="size-3" />,
  },
};

export function TaskStatusPill({ status }: { status: TaskStatus }) {
  const meta = taskMeta[status];
  return (
    <StatusPill tone={meta.tone} icon={meta.icon}>
      {meta.label}
    </StatusPill>
  );
}

const runMeta: Record<CodeRunStatus, { label: string; tone: Tone; icon: ReactNode }> = {
  queued: { label: "Queued", tone: "neutral", icon: <CircleDashed className="size-3" /> },
  running: {
    label: "Running",
    tone: "running",
    icon: <Loader2 className="size-3 animate-spin" />,
  },
  succeeded: { label: "Succeeded", tone: "success", icon: <CheckCircle2 className="size-3" /> },
  failed: { label: "Failed", tone: "danger", icon: <XCircle className="size-3" /> },
  timed_out: { label: "Timed out", tone: "warning", icon: <CircleDashed className="size-3" /> },
};

export function RunStatusPill({ status }: { status: CodeRunStatus }) {
  const meta = runMeta[status];
  return (
    <StatusPill tone={meta.tone} icon={meta.icon}>
      {meta.label}
    </StatusPill>
  );
}

const activityMeta: Record<ActivityStatus, { label: string; tone: Tone }> = {
  ok: { label: "Completed", tone: "success" },
  denied: { label: "Not permitted", tone: "danger" },
  failed: { label: "Failed", tone: "danger" },
  pending: { label: "In progress", tone: "running" },
};

export function ActivityStatusPill({ status }: { status: ActivityStatus }) {
  const meta = activityMeta[status];
  return <StatusPill tone={meta.tone}>{meta.label}</StatusPill>;
}

export function ToolStatusIcon({ status }: { status: ToolStatus }) {
  if (status === "running")
    return <Loader2 className="size-3.5 animate-spin text-info" aria-hidden />;
  if (status === "succeeded")
    return <CheckCircle2 className="size-3.5 text-success" aria-hidden />;
  return <XCircle className="size-3.5 text-destructive" aria-hidden />;
}

const policyMeta: Record<PolicyDecision, { label: string; tone: Tone; icon: ReactNode }> = {
  allowed: { label: "Allowed", tone: "success", icon: <ShieldCheck className="size-3" /> },
  denied: { label: "Not permitted", tone: "danger", icon: <ShieldAlert className="size-3" /> },
  requires_approval: {
    label: "Needs your approval",
    tone: "warning",
    icon: <ShieldQuestion className="size-3" />,
  },
};

export function PolicyDecisionPill({ decision }: { decision: PolicyDecision }) {
  const meta = policyMeta[decision];
  return (
    <StatusPill tone={meta.tone} icon={meta.icon}>
      {meta.label}
    </StatusPill>
  );
}
