import { ShieldAlert, ShieldQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PolicyNotice as PolicyNoticeModel } from "@/services/types";
import { cn } from "@/lib/utils";

/**
 * Renders a policy decision made by the backend.
 *
 * The frontend never evaluates policy, never predicts an outcome, and never
 * shows internal rule names or configuration — only the user-safe explanation
 * the backend returned.
 */
export function PolicyNotice({
  notice,
  onApprove,
  onDecline,
  className,
}: {
  notice: PolicyNoticeModel;
  onApprove?: () => void;
  onDecline?: () => void;
  className?: string;
}) {
  if (notice.decision === "allowed") return null;
  const denied = notice.decision === "denied";

  return (
    <div
      role="status"
      className={cn(
        "rounded-lg border px-4 py-3",
        denied ? "border-destructive/40 bg-destructive/10" : "border-warning/40 bg-warning/10",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        {denied ? (
          <ShieldAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
        ) : (
          <ShieldQuestion className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
        )}
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-foreground">
            {denied ? "This action wasn't permitted" : "This action needs your approval"}
          </p>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{notice.reason}</p>
          {notice.approvable && !denied ? (
            <div className="mt-3 flex gap-2">
              <Button size="sm" onClick={onApprove}>
                Approve
              </Button>
              <Button size="sm" variant="outline" onClick={onDecline}>
                Decline
              </Button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
