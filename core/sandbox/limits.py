"""Resource limits and sandbox configuration.

The :class:`ResourceLimits` dataclass defines the configurable capacity
of a sandboxed execution. The :class:`SandboxConfig` ties those limits
together with the choice of backend and the security-critical
allowlist (environment variables, languages, image name, etc.).

The defaults are conservative for an AI coding sandbox. They can be
raised or lowered at the call-site, but **never beyond the configured
maximum** (see :class:`SandboxConfig`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


# Maximum code size the sandbox will accept from a caller. This is
# a defence-in-depth limit; the Docker backend also enforces its own
# hard limit via the image's filesystem size.
DEFAULT_MAX_CODE_BYTES = 64 * 1024  # 64 KiB


# Defaults for resource limits. These are conservative and intended to
# prevent a single malicious or buggy piece of generated code from
# consuming the host's resources.
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KiB per stream
DEFAULT_MAX_MEMORY_MB = 256
DEFAULT_MAX_CPUS = 1.0
DEFAULT_MAX_PIDS = 64
DEFAULT_MAX_WORKSPACE_BYTES = 16 * 1024 * 1024  # 16 MiB


# The set of environment variables the sandbox is allowed to inherit.
# Anything else from the host's environment MUST NOT be passed to the
# container. This is a critical security boundary.
DEFAULT_ALLOWED_ENV_VARS: Tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "TZ",
)


# The default sandbox image. Kept as a string so the configuration can
# be overridden without code changes.
DEFAULT_SANDBOX_IMAGE = "sovereign-ai/sandbox-python:5d"


# Languages accepted by the sandbox. Currently Python only.
DEFAULT_SUPPORTED_LANGUAGES: Tuple[str, ...] = ("python",)


# ---------------------------------------------------------------------------
# ResourceLimits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceLimits:
    """Resource limits applied to a single sandboxed execution.

    These are *hard* upper bounds. The sandbox enforces them at the
    container/runtime level; the Python code inside the container
    cannot raise them. They are also upper bounds on the per-request
    overrides a caller can supply.
    """

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Maximum wall-clock execution time, in seconds."""

    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    """Maximum bytes captured per output stream (stdout, stderr).
    Exceeding this is reported as :attr:`ExecutionResult.output_truncated`."""

    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    """Maximum resident memory available to the container, in MiB."""

    max_cpus: float = DEFAULT_MAX_CPUS
    """Maximum CPU quota, in cores. 1.0 means one full core."""

    max_pids: int = DEFAULT_MAX_PIDS
    """Maximum number of processes (PIDs) the container may create."""

    max_workspace_bytes: int = DEFAULT_MAX_WORKSPACE_BYTES
    """Maximum total bytes that may be written into the sandbox workspace.
    Implemented as a temporary filesystem size limit."""

    max_code_bytes: int = DEFAULT_MAX_CODE_BYTES
    """Maximum size of the input code, in bytes (after UTF-8 encoding)."""

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_cpus <= 0:
            raise ValueError("max_cpus must be positive")
        if self.max_pids < 1:
            raise ValueError("max_pids must be at least 1")
        if self.max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive")
        if self.max_code_bytes <= 0:
            raise ValueError("max_code_bytes must be positive")


# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxConfig:
    """The complete configuration of a sandbox instance.

    Combines the resource limits with the security-critical settings:
    the chosen backend, the allowlisted environment variables, the
    supported languages, and the container image to use.

    The ``max_*`` fields here are absolute upper bounds. A
    :class:`ResourceLimits` or per-request override that tries to
    exceed them is clamped down to the maximum (or, depending on
    context, rejected outright).
    """

    limits: ResourceLimits = field(default_factory=ResourceLimits)
    """The default resource limits applied to every execution."""

    max_timeout_seconds: float = DEFAULT_MAX_TIMEOUT_SECONDS
    """Absolute maximum timeout, even if a request asks for more."""

    image: str = DEFAULT_SANDBOX_IMAGE
    """Container image to use for sandboxed Python execution."""

    allowed_env_vars: Tuple[str, ...] = DEFAULT_ALLOWED_ENV_VARS
    """Allowlist of environment variables that may be passed to the container.
    Anything else from the host's environment is NOT inherited."""

    supported_languages: Tuple[str, ...] = DEFAULT_SUPPORTED_LANGUAGES
    """The set of languages this sandbox accepts. Anything else is rejected."""

    network_enabled: bool = False
    """If True, the container is given network access. MUST be False for
    an air-gapped AI sandbox. This field exists for testing only —
    the default is the safe value."""

    read_only_root: bool = True
    """If True, the container's root filesystem is mounted read-only.
    A small tmpfs workspace is the only writable area."""

    drop_capabilities: bool = True
    """If True, all Linux capabilities are dropped. The container starts
    with the minimum capability set."""

    user: str = "sandbox"
    """The unprivileged user inside the container. Must NOT be ``root``."""

    def __post_init__(self) -> None:
        if self.max_timeout_seconds <= 0:
            raise ValueError("max_timeout_seconds must be positive")
        if self.max_timeout_seconds < self.limits.timeout_seconds:
            raise ValueError(
                "max_timeout_seconds must be at least the default timeout"
            )
        if not self.image:
            raise ValueError("image must be a non-empty string")
        if self.user == "root":
            # We explicitly forbid running as root inside the sandbox.
            raise ValueError("sandbox user must not be 'root'")
        if not self.user:
            raise ValueError("sandbox user must be a non-empty string")
        if not self.supported_languages:
            raise ValueError("supported_languages must be non-empty")

    def is_language_supported(self, language: str) -> bool:
        return language in self.supported_languages

    def effective_timeout(self, requested: float) -> float:
        """Clamp a requested timeout to ``[0, max_timeout_seconds]``."""
        if requested <= 0:
            return self.limits.timeout_seconds
        return min(requested, self.max_timeout_seconds)

    def effective_limits(
        self,
        overrides: Optional[dict],
    ) -> ResourceLimits:
        """Return a :class:`ResourceLimits` with overrides applied and clamped.

        Each override must be one of the fields of :class:`ResourceLimits`.
        Values that exceed the configured maximums are silently clamped.
        Negative or zero values are ignored.
        """
        if not overrides:
            return self.limits

        kwargs = {}
        for field_name in (
            "timeout_seconds",
            "max_output_bytes",
            "max_memory_mb",
            "max_cpus",
            "max_pids",
            "max_workspace_bytes",
            "max_code_bytes",
        ):
            if field_name not in overrides:
                continue
            value = overrides[field_name]
            if not isinstance(value, (int, float)):
                continue
            if value <= 0:
                continue
            kwargs[field_name] = value

        # Apply timeout clamping.
        if "timeout_seconds" in kwargs:
            kwargs["timeout_seconds"] = min(
                float(kwargs["timeout_seconds"]),
                self.max_timeout_seconds,
            )

        # Clamp other "max" fields to a conservative upper bound.
        # We do not have explicit "max" fields for these in this
        # config, but we use a hardcoded absolute ceiling to prevent
        # trivial bypass.
        for f, ceiling in (
            ("max_output_bytes", 10 * 1024 * 1024),       # 10 MiB
            ("max_memory_mb", 4 * 1024),                  # 4 GiB
            ("max_cpus", 8.0),
            ("max_pids", 1024),
            ("max_workspace_bytes", 1024 * 1024 * 1024),  # 1 GiB
            ("max_code_bytes", 1024 * 1024),              # 1 MiB
        ):
            if f in kwargs:
                kwargs[f] = min(kwargs[f], ceiling)

        if not kwargs:
            return self.limits
        return replace(self.limits, **kwargs)
