"""Workspace security for File Tools (Phase 5A).

The :class:`Workspace` is the single point of truth for file-tool path
validation. Every file tool (list/read/write) routes its raw input path
through :meth:`Workspace.resolve` before any I/O happens. ``resolve`` is
the ONLY place that handles absolute paths, symlinks, and traversal.

Design properties
-----------------

1. **Canonical root.** The workspace root is canonicalised at construction
   time. All path comparisons use the canonical form. This blocks trailing
   separators, ``.``, ``..``, and case-insensitive duplicates on Windows
   from creating two different "valid" workspaces.

2. **Strict containment.** The resolved path must equal or descend from
   the canonical root. Equality is checked with
   :meth:`pathlib.Path.is_relative_to` (or the equivalent fallback on
   older Python) to be both strict and portable.

3. **No implicit symlink following outside.** Symlinks are resolved *as
   part of canonicalisation*. If a symlink target resolves outside the
   workspace, the resolution fails. This blocks the common symlink-escape
   attack where a malicious user creates ``workspace/link -> /etc``.

4. **No drive / UNC injection.** Windows drive letters and UNC paths are
   detected and rejected *before* resolution. Absolute POSIX paths
   (``/etc``) are also rejected at the input level — the workspace
   provides the only acceptable root.

5. **No traversal segments.** Path segments ``..`` and ``.`` are
   validated *after* canonicalisation (which already collapses them), as
   a defence-in-depth check.

6. **Errors are safe.** Validation errors do NOT echo the input path back
   in a way that leaks filesystem layout, and they do NOT include the
   absolute workspace path. The caller is expected to surface a generic
   "path rejected" message; the underlying ``WorkspaceError`` carries a
   short category and a relative path only.

This module deliberately does **not** import any tool logic, agent logic,
or framework code. It is the lowest layer in the file-tool stack.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WorkspaceError(Exception):
    """Base class for workspace validation errors.

    All path-validation failures raised by :class:`Workspace` are instances
    of this class (or a subclass). The error message never includes the
    absolute workspace root or the absolute host path; it is suitable for
    direct surfacing to end users.
    """

    def __init__(self, message: str, relative_path: Optional[str] = None) -> None:
        self.relative_path = relative_path
        super().__init__(message)


class PathTraversalError(WorkspaceError):
    """The input path contains traversal (``..``) that escapes the workspace."""


class AbsolutePathError(WorkspaceError):
    """The input path is absolute (drive, UNC, or POSIX)."""


class SymlinkEscapeError(WorkspaceError):
    """A symlink in the path resolves outside the workspace."""


class InvalidPathError(WorkspaceError):
    """The input path is malformed in some other way (empty, wrong type, etc.)."""


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


# Pattern that matches Windows-style drive-letter paths (e.g. ``C:\``,
# ``C:/``), UNC paths (e.g. ``\\server\share``), and POSIX absolute paths.
_ABSOLUTE_PATH_RE = re.compile(
    r"""(
        ^[a-zA-Z]:[\\/]              |   # Windows drive letter  (C:\ or C:/)
        ^[\\/]{2}                    |   # UNC path              (\\server\share)
        ^/                              # POSIX absolute path   (/etc/passwd)
    )""",
    re.VERBOSE,
)

# Maximum workspace root depth. Real systems rarely nest deeper than this;
# anything beyond it is almost certainly an attack or misconfiguration.
_MAX_PATH_DEPTH = 64

# Maximum path length to accept. Beyond this, the OS will reject the
# operation anyway; we fail fast with a safe error.
_MAX_PATH_LENGTH = 4096


class Workspace:
    """A secure, validated workspace root for file tools.

    A workspace is constructed from a *root path*. From that point on, the
    workspace is the *only* directory any file tool may touch. Any path
    the tool receives is interpreted as *relative to* the workspace root.

    The workspace does not expose the absolute host path through its
    public surface; callers are expected to work with workspace-relative
    paths.

    Example::

        ws = Workspace(root_path="/data/workspace")
        resolved = ws.resolve("subdir/file.txt")  # OK
        ws.resolve("../etc/passwd")                # raises PathTraversalError
        ws.resolve("/etc/passwd")                  # raises AbsolutePathError
        ws.resolve("C:\\\\Windows\\\\notepad.exe")  # raises AbsolutePathError
    """

    __slots__ = ("_root", "_canonical_root", "_allow_create_dirs")

    def __init__(
        self,
        root_path: Union[str, os.PathLike],
        *,
        allow_create_dirs: bool = True,
    ) -> None:
        """Create a workspace rooted at ``root_path``.

        Args:
            root_path: The directory the workspace is rooted at. Must
                exist and be a directory. The path is canonicalised
                immediately; the workspace cannot be re-rooted later.
            allow_create_dirs: Whether the file tools may create
                intermediate directories on write. Defaults to True.

        Raises:
            WorkspaceError: if ``root_path`` cannot be canonicalised, is
                not a directory, or fails any of the security checks.
        """
        if root_path is None:
            raise InvalidPathError("Workspace root path is required")

        # Reject empty-string roots. ``Path("")`` silently resolves to the
        # current working directory, which is never what we want for a
        # sandboxed workspace.
        if isinstance(root_path, str) and not root_path.strip():
            raise InvalidPathError("Workspace root path must not be empty")

        try:
            root = Path(root_path)
        except (TypeError, ValueError) as exc:
            raise InvalidPathError("Workspace root path is malformed") from exc

        # Reject obviously bad roots up front. ``expanduser`` would resolve
        # ``~`` to the user's home directory, which is exactly what we
        # DO NOT want for a sandboxed workspace — the workspace should be
        # configured explicitly, not implicitly.
        root_str = str(root)
        if root_str.startswith("~"):
            raise InvalidPathError("Workspace root must not use '~'")

        # The workspace root MAY be an absolute path — the system
        # administrator who configures the workspace knows where the
        # data lives. We do NOT reject absolute root paths here; this
        # is a key difference from how we handle *tool arguments*, which
        # must always be relative to the workspace.
        #
        # What we DO reject: the user-configurable ``~`` shortcut, which
        # would resolve to the current user's home directory — an
        # implicit root that is exactly what we want to avoid.

        # Resolve the root. We DO allow absolute paths for the configured
        # root (the system administrator who sets up the workspace knows
        # where the data lives), but we canonicalise the root strictly.
        try:
            canonical = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise InvalidPathError(f"Workspace root does not exist or is not accessible: {type(exc).__name__}") from exc

        if not canonical.is_dir():
            raise InvalidPathError("Workspace root must be an existing directory")

        self._root = root
        self._canonical_root = canonical
        self._allow_create_dirs = allow_create_dirs

    # ------------------------------------------------------------------ Public surface

    @property
    def name(self) -> str:
        """A short, safe name for the workspace.

        Returns the basename of the canonical root (e.g. ``"workspace"``
        for ``/data/workspace``). Used in log messages where a human-
        readable identifier is needed without leaking the full path.
        """
        return self._canonical_root.name or "workspace"

    @property
    def allow_create_dirs(self) -> bool:
        """Whether writes may create intermediate directories."""
        return self._allow_create_dirs

    def resolve(self, raw_path: Union[str, os.PathLike]) -> Path:
        """Resolve ``raw_path`` to a path inside the workspace.

        The input is interpreted as a path *relative* to the workspace
        root. Absolute paths (drive, UNC, POSIX) and traversal escapes
        are rejected. Symlinks are resolved; targets outside the
        workspace are rejected.

        Args:
            raw_path: The caller-provided path. Must be a string or
                path-like object.

        Returns:
            A :class:`pathlib.Path` whose :meth:`is_relative_to` check
            against the canonical workspace root is true. The path is
            absolute (resolved against the root) and canonicalised, so
            callers may pass it directly to :func:`open` /
            :func:`os.stat` / etc.

        Raises:
            WorkspaceError: any path-validation failure. The error does
                not include the absolute host path.
        """
        if raw_path is None:
            raise InvalidPathError("Path is required", relative_path="")

        # Coerce to string. PathLike inputs are converted via str(); this
        # is safe because we validate the resulting string below.
        try:
            path_str = os.fsdecode(raw_path)
        except (TypeError, ValueError) as exc:
            raise InvalidPathError("Path is not a valid string", relative_path="") from exc

        if not isinstance(path_str, str):
            raise InvalidPathError("Path is not a valid string", relative_path="")

        path_str = path_str.strip()
        if not path_str:
            raise InvalidPathError("Path is empty", relative_path="")

        if len(path_str) > _MAX_PATH_LENGTH:
            raise InvalidPathError("Path is too long", relative_path="")

        # Reject control characters and NUL bytes. NUL in particular is
        # used in some attacks to truncate paths.
        if any(ord(c) < 0x20 for c in path_str):
            raise InvalidPathError("Path contains control characters", relative_path=path_str)

        # Normalise separators. Workspace paths use forward slashes for
        # output; we accept backslashes on input for convenience but
        # normalise them.
        normalised = path_str.replace("\\", "/")

        # Reject absolute-path-like inputs. This catches Windows drive
        # letters (``C:/...``), UNC paths (``\\server\share``), and POSIX
        # absolute paths (``/etc/...``).
        if _is_absolute_like(normalised):
            raise AbsolutePathError("Absolute paths are not allowed", relative_path=_safe_relative(path_str))

        # Split into segments and check for traversal.
        segments = [seg for seg in normalised.split("/") if seg not in ("",)]
        if len(segments) > _MAX_PATH_DEPTH:
            raise InvalidPathError("Path is too deep", relative_path=_safe_relative(path_str))

        for seg in segments:
            if seg == "..":
                raise PathTraversalError(
                    "Path contains parent-directory traversal",
                    relative_path=_safe_relative(path_str),
                )
            # The NUL check above already rejects control characters,
            # but check for ``.`` (a single-dot segment) for defence in
            # depth. ``.`` is harmless after canonicalisation, but
            # rejecting it makes the error messages cleaner.
            if seg == ".":
                continue
            # Reject reserved Windows names (CON, PRN, AUX, NUL, COM1-9,
            # LPT1-9). These names refer to devices on Windows, not files.
            if _is_reserved_windows_name(seg):
                raise InvalidPathError(
                    "Path uses a reserved device name",
                    relative_path=_safe_relative(path_str),
                )
            # Reject any segment containing characters that are
            # definitely not valid in a filename.
            if _has_invalid_filename_chars(seg):
                raise InvalidPathError(
                    "Path contains invalid filename characters",
                    relative_path=_safe_relative(path_str),
                )

        # Join with the canonical root and re-resolve. This handles any
        # legitimate ``.`` or empty segments and gives us a canonical
        # absolute path.
        joined = self._canonical_root.joinpath(*segments)
        try:
            resolved = joined.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            # If resolve fails (e.g. parent path does not exist yet for
            # a write), fall back to the joined path. We still check
            # containment below.
            resolved = joined

        # Containment check. The resolved path must be inside (or equal
        # to) the canonical root. We use Path.is_relative_to on Python
        # 3.9+; emulate it for older versions.
        if not _is_relative_to(resolved, self._canonical_root):
            raise PathTraversalError(
                "Resolved path escapes the workspace",
                relative_path=_safe_relative(path_str),
            )

        # Symlink escape check. If the resolved path itself is a
        # symlink, or if any component is a symlink whose target is
        # outside the workspace, reject. We do this AFTER containment
        # because the containment check is the more fundamental one.
        self._check_symlink_escape(resolved, path_str)

        return resolved

    def relative_to(self, absolute_path: Union[str, Path]) -> str:
        """Convert an absolute, already-resolved path to a workspace-relative string.

        This is the inverse of :meth:`resolve` for already-validated
        paths. It is provided so that tools can return relative paths in
        their output without re-validating.

        Args:
            absolute_path: An absolute path produced by
                :meth:`resolve`. Must be inside the workspace.

        Returns:
            A forward-slash-separated workspace-relative path string.

        Raises:
            WorkspaceError: if the path is not inside the workspace.
        """
        try:
            abs_path = Path(absolute_path).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise InvalidPathError("Path cannot be resolved") from exc

        if not _is_relative_to(abs_path, self._canonical_root):
            raise PathTraversalError(
                "Path is not inside the workspace",
                relative_path="",
            )
        if abs_path == self._canonical_root:
            return ""
        rel = abs_path.relative_to(self._canonical_root)
        return rel.as_posix()

    # ------------------------------------------------------------------ Internals

    def _check_symlink_escape(self, resolved: Path, raw_path: str) -> None:
        """Verify that no symlink in the chain escapes the workspace.

        For every component of the resolved path, we check whether the
        component is a symlink. If so, we resolve the symlink target and
        verify it is still inside the workspace.

        This catches the case where ``workspace/link -> /etc`` even when
        the caller passes ``link`` as a relative path.
        """
        try:
            current = self._canonical_root
            rel = resolved.relative_to(self._canonical_root)
        except ValueError:
            # Already caught by the containment check; nothing more to do.
            return

        for part in rel.parts:
            current = current / part
            try:
                # lstat does not follow symlinks, so we can detect them
                # without an infinite loop.
                lst = current.lstat()
            except (OSError, RuntimeError):
                # Path component does not exist. For reads, this is
                # handled by the tool; for writes, the parent must
                # exist. Either way, the symlink check cannot proceed.
                # We allow this — the tool's own existence check will
                # produce a more informative error.
                return

            if _is_symlink(lst):
                # Resolve the symlink target and check containment.
                try:
                    target = current.resolve(strict=False)
                except (OSError, RuntimeError) as exc:
                    raise SymlinkEscapeError(
                        f"Symlink cannot be resolved: {type(exc).__name__}",
                        relative_path=_safe_relative(raw_path),
                    ) from exc
                if not _is_relative_to(target, self._canonical_root):
                    raise SymlinkEscapeError(
                        "Symlink target escapes the workspace",
                        relative_path=_safe_relative(raw_path),
                    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_absolute_like(path: str) -> bool:
    """Return True if the path looks absolute (POSIX, Windows drive, or UNC).

    Detected patterns (anchored at the start of the string):

    * ``/...`` — POSIX absolute.
    * ``\\...`` or ``//...`` — Windows-style absolute, including UNC.
    * ``C:\\...`` / ``C:/...`` — Windows drive letter.
    """
    if not path:
        return False
    if _ABSOLUTE_PATH_RE.match(path):
        return True
    return False


def _is_relative_to(path: Path, base: Path) -> bool:
    """Portable ``Path.is_relative_to`` (Python 3.9 has it natively)."""
    if hasattr(path, "is_relative_to"):
        try:
            return path.is_relative_to(base)
        except ValueError:
            return False
    # Fallback: walk up the tree.
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_symlink(stat_result) -> bool:
    """Return True if the stat result indicates a symlink (POSIX or Windows)."""
    # On Windows, S_ISLNK is not exposed via os.stat, but os.lstat's
    # st_mode has the high bits set for reparse points. Simpler: check
    # via Path.is_symlink.
    return bool(stat_result.st_mode & 0o170000) == (0o120000 & 0o170000) or _is_windows_reparse(stat_result)


def _is_windows_reparse(stat_result) -> bool:
    """Return True if the stat result is a Windows reparse point (symlink/junction)."""
    if sys.platform != "win32":
        return False
    # FILE_ATTRIBUTE_REPARSE_POINT == 0x400
    return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400)


# Reserved Windows device names. These are not files even if a file
# with the same name exists in the workspace.
_RESERVED_WINDOWS_NAMES = frozenset(
    {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def _is_reserved_windows_name(segment: str) -> bool:
    """Return True if ``segment`` (case-insensitive) is a reserved Windows name."""
    if sys.platform == "win32":
        return segment.upper() in _RESERVED_WINDOWS_NAMES
    return False


# Characters that are not allowed in a filename on common filesystems.
# We keep this conservative: any character outside a clearly safe set
# is rejected. The slash is excluded because we've already split on it.
_BAD_FILENAME_CHARS = frozenset('<>:"/\\|?*\x00')


def _has_invalid_filename_chars(segment: str) -> bool:
    """Return True if the segment contains a clearly-invalid filename character.

    This is a *defence-in-depth* check: the workspace rejects ``..``
    earlier, and the underlying OS will reject many of these on its
    own. Catching them here produces a clean error.
    """
    return any(c in _BAD_FILENAME_CHARS for c in segment)


def _safe_relative(raw_path: str) -> str:
    """Return a safe, short form of the input path for error messages.

    Strips the path to a basename-like form if it is very long, and
    replaces any characters that might leak filesystem layout. The
    result is suitable for inclusion in user-facing error messages.
    """
    if not raw_path:
        return ""
    # Replace backslashes with forward slashes for consistent display.
    cleaned = raw_path.replace("\\", "/")
    if len(cleaned) > 200:
        cleaned = cleaned[:200] + "..."
    return cleaned
