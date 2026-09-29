/**
 * Frontend runtime configuration.
 *
 * The backend base URL is configurable; nothing is hardcoded to a host.
 * `USE_FIXTURES` is true until the backend API contracts are documented and
 * the real service adapters are implemented. Fixture data is always labelled
 * in the UI and never simulates a real authorization or policy decision.
 */
export const API_BASE_URL: string =
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined) ?? "";

export const USE_FIXTURES: boolean =
  import.meta.env["VITE_USE_FIXTURES"]?.toLowerCase().trim() !== "false";

console.log('[Config] USE_FIXTURES:', USE_FIXTURES);