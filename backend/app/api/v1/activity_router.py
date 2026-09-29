"""Activity-related routes for the v1 API.

Exposes endpoints for audit/activity events.
All operations go through the audit logger and policy engine.
"""
from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from core.agent.types import ToolCall
from core.security.policy_engine import get_policy_engine
from core.tools.workspace import Workspace

logger = logging.getLogger("sovereign-ai.api.activity")


def get_workspace() -> Workspace:
    """Return a workspace instance for file operations."""
    workspace_path = settings.workspace_path or "workspace"
    return Workspace(workspace_path)


class ActivityEventItem(BaseModel):
    id: str
    timestamp: str
    action: str
    status: str  # "ok" | "denied" | "failed" | "pending"
    detail: Optional[str] = None
    redactedFields: Optional[List[str]] = None


class ActivityResponse(BaseModel):
    items: List[ActivityEventItem]


router = APIRouter(prefix="/activity", tags=["activity"])


@router.get("", response_model=ActivityResponse)
async def list_activity() -> ActivityResponse:
    """List audit/activity events."""
    # Check policy for reading audit log
    policy_engine = get_policy_engine()
    if policy_engine is not None:
        tool_call = ToolCall(
            call_id="read_audit_log",
            tool_name="read_audit_log",
            arguments={},
        )
        policy_decision = policy_engine.evaluate(tool_call)
        if not policy_decision.is_allowed():
            logger.warning("Policy denied read_audit_log: %s", policy_decision.reason)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=policy_decision.reason,
            )

    # Determine audit log file path (same defaults as AuditLogger)
    log_dir = os.getenv("SOVEREIGN_AI_AUDIT_LOG_DIR", "data/audit")
    log_file = os.getenv("SOVEREIGN_AI_AUDIT_LOG_FILE", "audit.log")
    log_path = os.path.join(log_dir, log_file)

    if not os.path.exists(log_path):
        # Return empty list if log file does not exist yet
        return ActivityResponse(items=[])

    items: List[ActivityEventItem] = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line in audit log: %s", line)
                    continue

                # Map audit log entry to ActivityEventItem
                timestamp = event.get("timestamp", "")
                event_type = event.get("event_type", "unknown")
                outcome = event.get("outcome", "")

                # Map outcome to frontend status
                status_map = {
                    "ALLOW": "ok",
                    "SUCCESS": "ok",
                    "DENY": "denied",
                    "FAILURE": "failed",
                    "TIMEOUT": "failed",
                }
                event_status = status_map.get(outcome, "pending")

                # Action: use event_type
                action = event_type

                # Detail: try to extract a meaningful message from details
                detail = None
                details = event.get("details", {})
                if isinstance(details, dict):
                    # Prefer reason or error if present
                    if "reason" in details:
                        detail = str(details["reason"])
                    elif "error" in details:
                        detail = str(details["error"])
                    elif "latency_ms" in details:
                        detail = f"latency_ms: {details['latency_ms']}"
                    # If still no detail, we can leave it None

                # ID: use timestamp and line number to make it unique
                event_id = f"{timestamp}-{line_num}" if timestamp else str(line_num)

                items.append(
                    ActivityEventItem(
                        id=event_id,
                        timestamp=timestamp,
                        action=action,
                        status=event_status,
                        detail=detail,
                        # redactedFields omitted as we don't have that info
                    )
                )
    except Exception as exc:
        logger.exception("Failed to read audit log: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read audit log",
        )

    # Return items in reverse chronological order (newest first) so that the latest events are at the top
    items.reverse()
    return ActivityResponse(items=items)


# Export the router
__all__ = ["router"]