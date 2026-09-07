"""Audit Logger for SovereignAI Workbench.

Provides a centralized, framework-independent audit logging system that records
security-relevant events without logging sensitive data.

Design:
* Append-only log file with rotation based on size.
* Each log entry is a JSON line containing:
    - timestamp: ISO 8601 UTC timestamp
    - event_type: string identifier for the type of event
    - capability: capability string from policy engine (if applicable)
    - action: action string from policy engine (if applicable)
    - outcome: result of the event (e.g., ALLOW, DENY, SUCCESS, FAILURE)
    - correlation_id: identifier to trace related events (e.g., task_id, call_id)
    - details: optional non-sensitive contextual information
* Never logs sensitive data: file contents, prompts, credentials, etc.
* Logging failures do not affect primary operation (best-effort).
* Configurable retention via environment variables with safe defaults.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# Default configuration (can be overridden by environment variables)
DEFAULT_LOG_DIR = os.getenv("SOVEREIGN_AI_AUDIT_LOG_DIR", "data/audit")
DEFAULT_LOG_FILE = os.getenv("SOVEREIGN_AI_AUDIT_LOG_FILE", "audit.log")
DEFAULT_MAX_BYTES = int(os.getenv("SOVEREIGN_AI_AUDIT_MAX_BYTES", "5242880"))  # 5 MB
DEFAULT_BACKUP_COUNT = int(os.getenv("SOVEREIGN_AI_AUDIT_BACKUP_COUNT", "3"))
DEFAULT_ENCODING = "utf-8"

# Thread-safe singleton instance
_instance: Optional["AuditLogger"] = None
_instance_lock = threading.Lock()

#: Counter used to give every :class:`AuditLogger` instance a unique logger
#: name, so independent instances (e.g. in tests) never share handlers.
_instance_counter = 0
_instance_counter_lock = threading.Lock()


class AuditLogger:
    """Centralized audit logger for security-relevant events.

    The logger is thread-safe and designed for high-throughput scenarios.
    Log entries are written as JSON lines to a rotating file.
    """

    def __init__(
        self,
        log_dir: str = DEFAULT_LOG_DIR,
        log_file: str = DEFAULT_LOG_FILE,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        self._log_dir = log_dir
        self._log_file = log_file
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._encoding = encoding
        # Use a unique logger name per instance so independent instances
        # (e.g. multiple tests) never share handlers.
        global _instance_counter
        with _instance_counter_lock:
            _instance_counter += 1
            self._logger = logging.getLogger(f"sovereign-ai.audit.{_instance_counter}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        try:
            self._setup_handler()
        except Exception as exc:  # pragma: no cover - defensive
            # Never let audit logging break the caller. Fall back to a
            # stderr handler so events are still surfaced somewhere.
            self._logger.addHandler(logging.StreamHandler(sys.stderr))

    def _setup_handler(self) -> None:
        """Configure the rotating file handler."""
        os.makedirs(self._log_dir, exist_ok=True)
        log_path = self._get_log_path()
        handler = RotatingFileHandler(
            log_path,
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
            encoding=self._encoding,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def _get_log_path(self) -> str:
        """Return the full path to the log file."""
        return os.path.join(self._log_dir, self._log_file)

    #: Keys whose values are redacted from ``details`` (defence in depth).
    #: Sensitive data — file contents, prompts, model responses, credentials,
    #: tokens, OCR text, image bytes — must never reach the audit log, even if
    #: a caller accidentally passes them in ``details``.
    _SENSITIVE_KEY_TOKENS = frozenset(
        {
            "content", "code", "prompt", "response", "output", "stdin",
            "password", "passwd", "secret", "token", "access_token",
            "refresh_token", "api_key", "apikey", "authorization", "auth",
            "credential", "credentials", "private_key", "key", "stdout",
            "stderr", "image", "image_bytes", "image_data", "ocr", "ocr_text",
            "file_content", "text", "body",
        }
    )

    def _info(
        self,
        *,
        event_type: str,
        capability: Optional[str] = None,
        action: Optional[str] = None,
        outcome: str,
        correlation_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Internal method to log an audit event as JSON line.

        All fields are optional except event_type and outcome.
        ``details`` is sanitized (redacted + made JSON-safe) so that sensitive
        values never reach the log.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "outcome": outcome,
        }
        if capability is not None:
            event["capability"] = self._redact_scalar(capability)
        if action is not None:
            event["action"] = self._redact_scalar(action)
        if correlation_id is not None:
            event["correlation_id"] = self._redact_scalar(correlation_id)
        if details:
            event["details"] = self._sanitize(details)
        try:
            # Write as compact JSON line
            self._logger.info(json.dumps(event, separators=(",", ":")))
        except Exception:
            # Best-effort logging: if logging fails, we do not want to
            # disrupt the primary operation. Surface to stderr as a last resort.
            try:
                print(
                    f"AUDIT LOG FAILED: {json.dumps(event, separators=(',', ':'))}",
                    file=sys.stderr,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ sanitize

    def _sanitize(self, value: Any, _seen: Optional[set[int]] = None) -> Any:
        """Recursively redact sensitive keys and make the value JSON-safe.

        * Values under a sensitive key are replaced with ``"[REDACTED]"``.
        * Arbitrary objects are coerced to their string representation.
        * Bytes are never logged as raw bytes.
        * Overly long strings are truncated.
        * Cyclic references are broken (never recurse infinitely).
        """
        if _seen is None:
            _seen = set()
        if id(value) in _seen:
            return "[CYCLE]"
        _seen.add(id(value))
        try:
            if isinstance(value, dict):
                return {
                    str(k): (
                        "[REDACTED]"
                        if self._is_sensitive_key(str(k))
                        else self._sanitize(v, _seen)
                    )
                    for k, v in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [self._sanitize(v, _seen) for v in value]
            if isinstance(value, bytes):
                return "[BYTES]"
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                return self._redact_scalar(value)
            # Arbitrary objects: represent as a safe string.
            try:
                return self._redact_scalar(str(value))
            except Exception:
                return "[UNSERIALIZABLE]"
        finally:
            _seen.discard(id(value))

    def _redact_scalar(self, value: str) -> str:
        """Truncate overly long strings to a bounded length."""
        if not isinstance(value, str):
            return str(value)
        if len(value) > 512:
            return value[:512] + "...[TRUNCATED]"
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        """Return True if a details key is considered sensitive."""
        lowered = key.lower()
        return any(tok in lowered for tok in AuditLogger._SENSITIVE_KEY_TOKENS)

    def log_policy_decision(
        self,
        *,
        capability: str,
        action: str,
        decision: str,
        reason: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log a policy engine decision.

        Args:
            capability: The capability evaluated (e.g., "file_system")
            action: The action evaluated (e.g., "read_file")
            decision: The decision string (ALLOW, DENY, REQUIRE_APPROVAL)
            reason: The reason string from the policy engine (non-sensitive)
            correlation_id: Optional identifier to correlate with agent/tool calls
        """
        self._info(
            event_type="policy_decision",
            capability=capability,
            action=action,
            outcome=decision,
            correlation_id=correlation_id,
            details={"reason": reason},
        )

    def log_tool_execution_start(
        self,
        *,
        tool_name: str,
        correlation_id: str,
    ) -> None:
        """Log the start of a tool execution.

        Args:
            tool_name: Name of the tool being executed
            correlation_id: Identifier to correlate with the policy decision and completion
        """
        # We log the tool name as both capability and action for simplicity
        # since tools may not map directly to policy capability/action.
        self._info(
            event_type="tool_execution_start",
            capability=tool_name,
            action="start",
            outcome="STARTED",
            correlation_id=correlation_id,
        )

    def log_tool_execution_end(
        self,
        *,
        tool_name: str,
        correlation_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log the end of a tool execution.

        Args:
            tool_name: Name of the tool that was executed
            correlation_id: Identifier to correlate with the start event
            success: Whether the tool execution succeeded
            latency_ms: Optional latency in milliseconds
            error: Optional error message (must be non-sensitive, e.g., without file paths or credentials)
        """
        outcome = "SUCCESS" if success else "FAILURE"
        details: dict[str, Any] = {}
        if latency_ms is not None:
            details["latency_ms"] = latency_ms
        if error is not None:
            # Error message should already be sanitized by the caller
            details["error"] = error
        self._info(
            event_type="tool_execution_end",
            capability=tool_name,
            action="end",
            outcome=outcome,
            correlation_id=correlation_id,
            details=details if details else None,
        )

    def log_sandbox_execution(
        self,
        *,
        correlation_id: str,
        success: bool,
        output_length: Optional[int] = None,
        error: Optional[str] = None,
        timeout: bool = False,
    ) -> None:
        """Log a sandbox execution result.

        Args:
            correlation_id: Identifier to correlate with the tool execution
            success: Whether the sandbox execution succeeded
            output_length: Optional length of the output (if any) - never the content
            error: Optional error message (non-sensitive)
            timeout: Whether the execution timed out
        """
        if timeout:
            outcome = "TIMEOUT"
        else:
            outcome = "SUCCESS" if success else "FAILURE"
        details: dict[str, Any] = {}
        if output_length is not None:
            details["output_length"] = output_length
        if error is not None:
            details["error"] = error
        self._info(
            event_type="sandbox_execution",
            capability="sandbox",
            action="execution",
            outcome=outcome,
            correlation_id=correlation_id,
            details=details if details else None,
        )

    def log_file_operation(
        self,
        *,
        operation: str,
        correlation_id: str,
        success: bool,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a file operation (read/write/list) without logging the path itself.

        Args:
            operation: The operation performed (e.g., "read", "write", "list")
            correlation_id: Identifier to correlate with the tool execution
            success: Whether the operation succeeded
            details: Optional non-sensitive details (e.g., number of entries, size, etc.)
        """
        outcome = "SUCCESS" if success else "FAILURE"
        self._info(
            event_type="file_operation",
            capability="file_system",
            action=operation,
            outcome=outcome,
            correlation_id=correlation_id,
            details=details,
        )


def get_audit_logger() -> AuditLogger:
    """Return the process-wide audit logger instance.

    Creates the instance on first call (thread-safe singleton).
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AuditLogger()
    return _instance


def reset_audit_logger() -> None:
    """Reset the audit logger instance (primarily for testing)."""
    global _instance
    with _instance_lock:
        _instance = None