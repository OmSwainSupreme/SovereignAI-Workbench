import { USE_FIXTURES } from "@/lib/config";
import type { TaskStreamAdapter } from "./adapter";
import { fixtureStreamAdapter } from "./fixtureAdapter";
import { pollingStreamAdapter } from "./pollingAdapter";

/**
 * Adapter selection happens here and nowhere else.
 *
 * With fixtures enabled the sample adapter replays events for development and
 * tests. Otherwise the real adapter drives the backend's submit-then-poll task
 * contract. No component imports a concrete adapter.
 */
export function getTaskStreamAdapter(): TaskStreamAdapter {
  if (USE_FIXTURES) return fixtureStreamAdapter;
  return pollingStreamAdapter;
}

export type { TaskStreamAdapter } from "./adapter";
