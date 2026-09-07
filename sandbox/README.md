# SovereignAI Workbench — Sandbox

Isolated runtime environment for executing untrusted or user-supplied code safely.

## Purpose

The sandbox provides a secure execution environment for code submitted through the SovereignAI Workbench — for example, generated Python snippets, shell scripts, or plugin payloads. Isolation is essential to maintain the confidentiality guarantees of the air-gapped platform.

## Planned Capabilities

| Feature | Description |
|---|---|
| Python sandbox | Run generated Python code in a restricted subprocess or container |
| Shell sandbox | Execute shell commands with resource limits |
| Plugin sandbox | Load and execute third-party plugins with controlled permissions |
| Network isolation | Block all outbound network calls from sandboxed workloads |
| Resource limits | CPU time, memory, and disk I/O constraints to prevent DoS |
| Execution audit trail | Capture and log every sandboxed execution for the audit system |

## Security Goals

- Zero external network access from sandboxed workloads.
- No access to the host filesystem outside of designated scratch paths.
- Sandboxed code must never be able to read host credentials, environment secrets, or model weights.
- A tamper-evident audit trail records every execution (command, user, duration, exit code, output).

## Technology Options Under Evaluation

| Approach | Notes |
|---|---|
| **gVisor** | User-space kernel isolation for containers — strong separation, no host kernel shared |
| **seccomp + AppArmor** | System-call filtering on the host — lightweight, requires host-specific tuning |
| **subprocess + setrlimit** | Per-process resource limits — suitable for development only |
| **Docker-in-Docker** | Nested container isolation — requires `SYS_ADMIN` capability |

The final choice will depend on the deployment environment and the security posture required by the target organization.

## Current State

This directory is intentionally empty during the bootstrap phase. The sandbox architecture will be designed and implemented when the code-execution feature is reached in the development roadmap.

## Constraints

- The sandbox must function entirely offline — no outbound network calls are permitted.
- Sandboxed workloads must not access host credentials or secrets.
- Execution logs must be captured for the audit trail.
