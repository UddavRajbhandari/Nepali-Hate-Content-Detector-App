/**
 * API client — all backend calls in one place.
 *
 * Dev:  Vite proxy forwards /api/* to http://localhost:8000 (see vite.config.ts)
 * Prod: Set VITE_API_BASE=https://your-backend.railway.app/api in Vercel env vars
 */

const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");
const ROOT = BASE.replace(/\/api$/, "");

// Re-export Label from labels.ts — single source of truth
export type { Label } from "./labels";

// ---------------------------------------------------------------------------
// Types — exact mirror of backend/app/models/schemas.py + API_REFERENCE.md
// ---------------------------------------------------------------------------

export interface EmojiFeatures {
  has_hate_emoji: number;
  has_mockery_emoji: number;
  has_positive_emoji: number;
  has_sadness_emoji: number;
  has_fear_emoji: number;
  has_disgust_emoji: number;
  hate_emoji_count: number;
  mockery_emoji_count: number;
  positive_emoji_count: number;
  sadness_emoji_count: number;
  fear_emoji_count: number;
  disgust_emoji_count: number;
  total_emoji_count: number;
  hate_to_positive_ratio: number;
  has_mixed_sentiment: number;
  unknown_emoji_count: number;
  has_unknown_emoji: number;
  known_emoji_ratio: number;
}

export interface ScriptInfo {
  script_type: "devanagari" | "romanized_nepali" | "english" | "mixed" | "other";
  confidence: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

/** GET /api/status — which optional XAI packages are installed */
export interface StatusResponse {
  model_loaded: boolean;
  device: string;
  preprocessor: boolean;
  lime: boolean;
  shap: boolean;
  captum: boolean;
}

export interface PredictResult {
  prediction: import("./labels").Label;
  confidence: number;
  probabilities: Record<string, number>;
  original_text: string;
  preprocessed_text: string;
  emoji_features: EmojiFeatures;
  script_info: ScriptInfo | null;
  error: string | null;
}

export interface EmojiInfo {
  emojis_found: string[];
  total_count: number;
  known_emojis: string[];
  known_count: number;
  unknown_emojis: string[];
  unknown_count: number;
  coverage: number;
}

export interface AnalyzeResponse {
  script_info: ScriptInfo;
  emoji_info: EmojiInfo;
}

export interface WordScore {
  word: string;
  score: number;
}

export interface ExplainResult {
  method: "LIME" | "SHAP" | "Captum-IG";
  prediction: import("./labels").Label;
  confidence: number;
  word_scores: WordScore[];
  preprocessed_text: string;
  convergence_delta: number | null;
  error: string | null;
}

export interface BatchResultItem {
  text: string;
  full_text: string;
  prediction: string;
  confidence: number;
  preprocessed_text: string;
  error?: string;
}

export interface HistoryItem {
  timestamp: string;
  text: string;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  preprocessed_text: string;
  emoji_features: EmojiFeatures;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface HistoryStatsResponse {
  total: number;
  avg_confidence: number | null;
  class_counts: Record<string, number>;
  most_common_class: string | null;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function _post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail)
    );
  }
  return res.json() as Promise<T>;
}

async function _get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : res.statusText);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export const api = {
  // ── Boot ─────────────────────────────────────────────────────────────────
  /** Poll until model_loaded === true before unlocking the UI */
  health: () =>
    fetch(`${ROOT}/health`)
      .then((r) => r.json())
      .then((d) => d as HealthResponse),

  /** Call once after health — determines which XAI buttons to show */
  status: () => _get<StatusResponse>("/status"),

  // ── Predict ───────────────────────────────────────────────────────────────
  predict: (text: string, save = true) =>
    _post<PredictResult>("/predict", { text, save_to_history: save }),

  /** Lightweight preprocessing info — no model inference */
  analyze: (text: string) =>
    _post<AnalyzeResponse>("/analyze", { text }),

  // ── Explainability ────────────────────────────────────────────────────────
  // num_samples default 200 per API spec (was wrongly 300 before)
  explainLime: (text: string, num_samples = 200) =>
    _post<ExplainResult>("/explain/lime", { text, num_samples }),

  explainShap: (text: string) =>
    _post<ExplainResult>("/explain/shap", { text }),

  explainCaptum: (text: string, n_steps = 50) =>
    _post<ExplainResult>("/explain/captum", { text, n_steps }),

  // ── Batch (streaming NDJSON) ──────────────────────────────────────────────
  batchStream: async (
    texts: string[],
    onItem: (item: BatchResultItem, index: number, total: number) => void,
    onDone: (total: number) => void,
    signal?: AbortSignal
  ): Promise<void> => {
    const res = await fetch(`${BASE}/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? `Batch failed: ${res.statusText}`);
    }
    if (!res.body) throw new Error("No response body — streaming not supported.");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line) as
              | { done: true; total: number }
              | { index: number; total: number; result: BatchResultItem };
            if ("done" in msg && msg.done) {
              onDone(msg.total);
            } else if ("result" in msg) {
              onItem(msg.result, msg.index, msg.total);
              // Yield to the browser event loop after every item so it can repaint.
              // Without this, when the backend sends multiple results in one TCP
              // chunk the for-loop runs synchronously — flushSync updates React state
              // but the browser only paints when JS yields, so the counter jumps
              // straight to the final value instead of incrementing visibly.
              await new Promise<void>((r) => setTimeout(r, 0));
            }
          } catch {
            console.warn("Skipped malformed NDJSON line:", line);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  },

  // ── History ───────────────────────────────────────────────────────────────
  getHistory: (limit = 20, offset = 0) =>
    _get<HistoryResponse>(`/history?limit=${limit}&offset=${offset}`),

  getHistoryStats: () =>
    _get<HistoryStatsResponse>("/history/stats"),

  clearHistory: () =>
    fetch(`${BASE}/history`, { method: "DELETE" })
      .then((r) => r.json())
      .then((d) => d as { message: string; deleted_count: number }),
};