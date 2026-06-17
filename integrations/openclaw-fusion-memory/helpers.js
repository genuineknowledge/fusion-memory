export const DEFAULT_BASE_URL = "http://127.0.0.1:8765";
export const DEFAULT_TIMEOUT_MS = 1500;

export function normalizeBaseUrl(value) {
  return String(value || DEFAULT_BASE_URL).replace(/\/+$/, "");
}

export function safeFailure(_error) {
  return {
    content: [
      {
        type: "text",
        text: "Fusion Memory is not available. Continue without memory, then run fusion-memory doctor.",
      },
    ],
  };
}
