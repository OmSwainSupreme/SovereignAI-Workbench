import type { StreamEvent } from "../types";

/**
 * Task streaming adapter boundary.
 *
 * The backend's streaming transport is UNKNOWN until confirmed. The UI depends
 * on this interface and on the normalized `StreamEvent` union only, so the
 * transport (SSE, chunked fetch, websocket, or polling) can be swapped without
 * touching a single component.
 *
 * Backend event names are mapped into `StreamEvent` inside an adapter, never
 * in the UI.
 */
export interface StreamStartOptions {
  signal: AbortSignal;
  prompt: string;
  attachments?: string[];
}

export interface TaskStreamAdapter {
  readonly name: string;
  start(taskId: string, options: StreamStartOptions): AsyncIterable<StreamEvent>;
}
