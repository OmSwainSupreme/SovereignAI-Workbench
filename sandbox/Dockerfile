# SovereignAI Workbench — Sandboxed Python execution image (Phase 5D)
#
# This image is the runtime used by DockerSandbox. It runs untrusted
# Python code with the absolute minimum surface area.
#
# Build:
#   docker build -t sovereign-ai/sandbox-python:5d sandbox/
#
# Run (for manual inspection; do NOT use in production — the production
# invocation goes through DockerSandbox):
#   docker run --rm --network=none --read-only --user sandbox \
#       --tmpfs /workspace:size=16m,uid=1000,gid=1000 \
#       --cap-drop=ALL --security-opt=no-new-privileges \
#       sovereign-ai/sandbox-python:5d
#
# Security properties (intentional):
#   * No network tools installed (no curl, no wget, no pip).
#   * No package manager available at runtime.
#   * No shell utilities beyond busybox (or sh -c) for the entrypoint.
#   * Runs as a non-root user (uid 1000).
#   * Read-only root filesystem; /workspace is the only writable area.
#   * No host filesystem mounted.
#   * No Docker socket.
#   * No environment secrets.

FROM python:3.11-alpine

# We create an unprivileged user with a fixed uid/gid. The container
# is run with --user sandbox; the host uid is irrelevant.
RUN addgroup -g 1000 -S sandbox \
    && adduser  -u 1000 -S -G sandbox sandbox

# /workspace is the only writable directory. The sandbox's docker run
# invocation mounts a tmpfs there.
RUN mkdir -p /workspace && chown sandbox:sandbox /workspace

# Copy the entrypoint. This is the ONLY application code in the image;
# we do NOT copy the repository.
COPY entrypoint.py /usr/local/bin/sandbox_entrypoint.py
RUN chmod 0755 /usr/local/bin/sandbox_entrypoint.py

# Drop capabilities / set sane defaults. These are belt-and-braces;
# the actual hardening is applied via `docker run` flags by
# DockerSandbox, which is the source of truth.
#
# We do not install su-exec or tini; the entrypoint is small enough
# that we do not need them.

USER sandbox
WORKDIR /workspace

# Disable bytecode generation and force unbuffered I/O. These are
# defaults; the DockerSandbox passes them again as env vars, but
# having them baked in is safe.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random

# Set a safe PATH. We deliberately do not add any user-installed
# directories to PATH.
ENV PATH=/usr/local/bin:/usr/bin:/bin

# The entrypoint. The container takes a base64-encoded code blob from
# the SOVEREIGN_CODE_B64 env var, decodes it to /workspace/main.py,
# and runs it under python3 with the configured timeout and output
# cap.
ENTRYPOINT ["python3", "/usr/local/bin/sandbox_entrypoint.py"]
