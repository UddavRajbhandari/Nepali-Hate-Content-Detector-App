# Nepali Hate Content Detection — API Reference

> **Base URL:** `http://localhost:8000`  
> **Interactive docs:** `http://localhost:8000/docs` (Swagger UI)  
> **Start server:** `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000` from `major_project/` root

---

## Table of Contents

1. [Health Check](#1-health-check)
2. [Status & Capabilities](#2-status--capabilities)
3. [Predict — Single Text](#3-predict--single-text)
4. [Analyze — Preprocessing Info](#4-analyze--preprocessing-info)
5. [Explain — LIME](#5-explain--lime)
6. [Explain — SHAP](#6-explain--shap)
7. [Explain — Captum IG](#7-explain--captum-ig)
8. [Batch Predict (Streaming)](#8-batch-predict-streaming)
9. [History — Fetch](#9-history--fetch)
10. [History — Stats](#10-history--stats)
11. [History — Clear](#11-history--clear)
12. [Error Reference](#12-error-reference)
13. [TypeScript Types](#13-typescript-types)
14. [Frontend Integration Guide](#14-frontend-integration-guide)

---

## 1. Health Check

```
GET /health
```

Returns whether the server is up and the model has finished loading. Call this on app mount to gate the UI.

**Response `200`**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | `string` | Always `"ok"` if server is running |
| `model_loaded` | `boolean` | `false` while model is still downloading/loading at startup |
| `device` | `string` | `"cpu"` or `"cuda"` |

**Frontend use:** Poll this every 2 seconds on mount until `model_loaded === true`, then unlock the main UI.

---

## 2. Status & Capabilities

```
GET /api/status
```

Returns which optional XAI packages are installed. Call once on load to decide which Explain buttons to show or hide.

**Response `200`**
```json
{
  "model_loaded": true,
  "device": "cpu",
  "preprocessor": true,
  "lime": true,
  "shap": true,
  "captum": false
}
```

| Field | Type | Notes |
|---|---|---|
| `model_loaded` | `boolean` | Same as `/health` |
| `device` | `string` | `"cpu"` or `"cuda"` |
| `preprocessor` | `boolean` | If `false`, raw text is passed to model without script conversion |
| `lime` | `boolean` | Whether `lime` package is installed |
| `shap` | `boolean` | Whether `shap` package is installed |
| `captum` | `boolean` | Whether `captum` package is installed |

**Frontend use:** If `captum === false`, disable the Captum tab. Same for LIME/SHAP.

---

## 3. Predict — Single Text

```
POST /api/predict
Content-Type: application/json
```

Core endpoint. Preprocesses input → runs XLM-RoBERTa-large → returns label + probabilities + preprocessing details.

**Request body**
```json
{
  "text": "महिलाले घरमा बस्नु पर्छ",
  "save_to_history": true
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | `string` | ✅ | 1–5000 chars. Devanagari, Romanized Nepali, English, or mixed. Must not be whitespace only |
| `save_to_history` | `boolean` | ❌ | Default `true`. Saves result to `data/prediction_history.jsonl` as background task |

**Response `200`**
```json
{
  "prediction": "OS",
  "confidence": 0.9909,
  "probabilities": {
    "NO": 0.0034,
    "OO": 0.0041,
    "OR": 0.0016,
    "OS": 0.9909
  },
  "original_text": "महिलाले घरमा बस्नु पर्छ",
  "preprocessed_text": "महिलाले घरमा बस्नु पर्छ",
  "emoji_features": {
    "has_hate_emoji": 0,
    "has_mockery_emoji": 0,
    "has_positive_emoji": 0,
    "has_sadness_emoji": 0,
    "has_fear_emoji": 0,
    "has_disgust_emoji": 0,
    "hate_emoji_count": 0,
    "mockery_emoji_count": 0,
    "positive_emoji_count": 0,
    "sadness_emoji_count": 0,
    "fear_emoji_count": 0,
    "disgust_emoji_count": 0,
    "total_emoji_count": 0,
    "hate_to_positive_ratio": 0.0,
    "has_mixed_sentiment": 0,
    "unknown_emoji_count": 0,
    "has_unknown_emoji": 0,
    "known_emoji_ratio": 1.0
  },
  "script_info": {
    "script_type": "devanagari",
    "confidence": 0.98
  },
  "error": null
}
```

**Prediction labels**

| Label | Meaning | Display color |
|---|---|---|
| `NO` | Non-offensive | Green `#28a745` |
| `OO` | Other-offensive (general) | Yellow `#ffc107` |
| `OR` | Offensive-Racist (race/ethnicity/religion hate) | Red `#dc3545` |
| `OS` | Offensive-Sexist (gender/sexuality hate) | Purple `#6f42c1` |

**`emoji_features` fields**

18 fields total. All are `int` except `hate_to_positive_ratio` and `known_emoji_ratio` which are `float`.

| Field | Description |
|---|---|
| `has_hate_emoji` | Binary flag: 1 if text contains anger/weapon emojis |
| `hate_emoji_count` | Count of hate-related emojis |
| `has_positive_emoji` | Binary flag |
| `positive_emoji_count` | Count |
| `total_emoji_count` | Total emoji count |
| `hate_to_positive_ratio` | `hate_count / max(positive_count, 1)` |
| `has_mixed_sentiment` | 1 if both hate and positive emojis present |
| `unknown_emoji_count` | Emojis not in the mapping dictionary |
| `known_emoji_ratio` | Fraction of emojis that have Nepali translations |

**`script_info` fields**

| Field | Description |
|---|---|
| `script_type` | One of: `devanagari`, `romanized_nepali`, `english`, `mixed`, `other` |
| `confidence` | Float 0–1 |

**Error cases**

| Status | Condition |
|---|---|
| `422` | Empty or whitespace-only text |
| `503` | Model not yet loaded |
| `503` | Out of memory during inference |
| `500` | Unexpected server error |

---

## 4. Analyze — Preprocessing Info

```
POST /api/analyze
Content-Type: application/json
```

Lightweight endpoint — runs only script detection and emoji analysis, does **not** run the model. Use for the preprocessing details panel without triggering a full prediction.

**Request body**
```json
{
  "text": "timi murkha chau 😡"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | `string` | ✅ | 1–5000 chars |

**Response `200`**
```json
{
  "script_info": {
    "script_type": "romanized_nepali",
    "confidence": 0.80
  },
  "emoji_info": {
    "emojis_found": ["😡"],
    "total_count": 1,
    "known_emojis": ["😡"],
    "known_count": 1,
    "unknown_emojis": [],
    "unknown_count": 0,
    "coverage": 1.0
  }
}
```

**`emoji_info` fields**

| Field | Type | Description |
|---|---|---|
| `emojis_found` | `string[]` | All emoji characters found in text |
| `total_count` | `number` | Total emoji count |
| `known_emojis` | `string[]` | Emojis that have a Nepali translation mapping |
| `known_count` | `number` | |
| `unknown_emojis` | `string[]` | Emojis not in the mapping dictionary |
| `unknown_count` | `number` | |
| `coverage` | `number` | `known_count / total_count`, or `1.0` if no emojis |

---

## 5. Explain — LIME

```
POST /api/explain/lime
Content-Type: application/json
```

Generates word-level importance scores using LIME (Local Interpretable Model-agnostic Explanations). LIME perturbs the **preprocessed** text, so token labels always align with what the model saw.

**Request body**
```json
{
  "text": "महिलाले घरमा बस्नु पर्छ",
  "num_samples": 200,
  "n_steps": 50
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | `string` | ✅ | — | 1–2000 chars (shorter limit than predict — LIME runs many model calls) |
| `num_samples` | `integer` | ❌ | `200` | Range 50–1000. Higher = more reliable scores, higher latency |
| `n_steps` | `integer` | ❌ | `50` | Only used by Captum, ignored here |

**Response `200`**
```json
{
  "method": "LIME",
  "prediction": "OS",
  "confidence": 0.9909,
  "word_scores": [
    { "word": "घरमा", "score": 0.182 },
    { "word": "महिलाले", "score": 0.143 },
    { "word": "बस्नु", "score": 0.091 },
    { "word": "पर्छ", "score": -0.034 }
  ],
  "preprocessed_text": "महिलाले घरमा बस्नु पर्छ",
  "convergence_delta": null,
  "error": null
}
```

**`word_scores` interpretation**

| Score | Meaning |
|---|---|
| Positive | Word pushes prediction **toward** the predicted class |
| Negative | Word pushes prediction **away** from the predicted class |
| High absolute value | Strong influence |

Words are returned in LIME's natural order (by score magnitude). Sort by `abs(score)` descending for a ranked importance bar chart.

**Frontend rendering:** Horizontal bar chart. Positive bars green, negative bars red. Display `word` on the y-axis.

---

## 6. Explain — SHAP

```
POST /api/explain/shap
Content-Type: application/json
```

Generates attributions using SHAP. Falls back to leave-one-out occlusion if the primary SHAP text masker fails.

**Request body** — same shape as LIME. `num_samples` is ignored; `n_steps` is ignored.

```json
{
  "text": "महिलाले घरमा बस्नु पर्छ"
}
```

**Response `200`**
```json
{
  "method": "SHAP",
  "prediction": "OS",
  "confidence": 0.9909,
  "word_scores": [
    { "word": "घरमा", "score": 0.211 },
    { "word": "महिलाले", "score": 0.178 },
    { "word": "बस्नु", "score": 0.095 },
    { "word": "पर्छ", "score": -0.021 }
  ],
  "preprocessed_text": "महिलाले घरमा बस्नु पर्छ",
  "convergence_delta": null,
  "error": null
}
```

Word scores are **sorted by descending absolute value** — most influential words first.

If the fallback was used, `error` will be `"Used gradient_fallback"` (not a failure — result is still valid).

---

## 7. Explain — Captum IG

```
POST /api/explain/captum
Content-Type: application/json
```

Generates subword token attributions using Layer Integrated Gradients (Captum). Works at the subword tokenizer level, so words may appear as `▁महिलाले` (SentencePiece prefix).

**Request body**
```json
{
  "text": "महिलाले घरमा बस्नु पर्छ",
  "n_steps": 50
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `text` | `string` | ✅ | — | 1–2000 chars |
| `n_steps` | `integer` | ❌ | `50` | Range 10–200. Increase to 100+ if `convergence_delta > 0.05` |
| `num_samples` | `integer` | ❌ | `200` | Only used by LIME, ignored here |

**Response `200`**
```json
{
  "method": "Captum-IG",
  "prediction": "OS",
  "confidence": 0.9909,
  "word_scores": [
    { "word": "महिलाले", "score": 0.842 },
    { "word": "घरमा", "score": 0.631 },
    { "word": "बस्नु", "score": 0.417 },
    { "word": "पर्छ", "score": 0.203 }
  ],
  "preprocessed_text": "महिलाले घरमा बस्नु पर्छ",
  "convergence_delta": 0.0031,
  "error": null
}
```

| Field | Notes |
|---|---|
| `word_scores[].score` | Signed attribution (sum of subword attributions). Positive = contributes to prediction |
| `convergence_delta` | Quality indicator. Values below `0.05` = reliable. Increase `n_steps` if high |

**⚠️ Memory warning:** Captum is the most memory-intensive method. It may return `422` on low-RAM cloud deployments. Use LIME or SHAP as fallback — the frontend should check `captum` in `/api/status` before showing this option.

---

## 8. Batch Predict (Streaming)

```
POST /api/batch
Content-Type: application/json
```

Classifies multiple texts and **streams results back as NDJSON** (Newline-Delimited JSON). Each text is processed independently — an error on one does not abort the batch.

**Request body**
```json
{
  "texts": [
    "यो राम्रो छ",
    "तिमी मुर्ख हौ",
    "timi murkha chau"
  ]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `texts` | `string[]` | ✅ | 1–200 items. Empty strings are stripped silently |

**Response — NDJSON stream**

`Content-Type: application/x-ndjson`

Each line is a complete JSON object. Two types of lines:

**Progress line** (one per text):
```json
{
  "index": 0,
  "total": 3,
  "result": {
    "text": "यो राम्रो छ",
    "full_text": "यो राम्रो छ",
    "prediction": "NO",
    "confidence": 0.9721,
    "preprocessed_text": "यो राम्रो छ"
  }
}
```

**Final sentinel line** (last line always):
```json
{ "done": true, "total": 3 }
```

**Error result** (when one text fails):
```json
{
  "index": 1,
  "total": 3,
  "result": {
    "text": "...",
    "full_text": "...",
    "prediction": "Error",
    "confidence": 0.0,
    "preprocessed_text": "",
    "error": "error message"
  }
}
```

**Frontend streaming example (fetch API):**
```javascript
const response = await fetch("http://localhost:8000/api/batch", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ texts }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split("\n");
  buffer = lines.pop(); // keep incomplete last line

  for (const line of lines) {
    if (!line.trim()) continue;
    const data = JSON.parse(line);

    if (data.done) {
      // Batch complete
      setProgress(100);
    } else {
      // Update progress bar and results table
      setProgress(Math.round(((data.index + 1) / data.total) * 100));
      appendResult(data.result);
    }
  }
}
```

---

## 9. History — Fetch

```
GET /api/history?limit=100&offset=0
```

Returns saved predictions in reverse-chronological order (newest first).

**Query parameters**

| Param | Type | Default | Range | Description |
|---|---|---|---|---|
| `limit` | `integer` | `100` | 1–500 | Max records to return |
| `offset` | `integer` | `0` | ≥0 | Skip this many records from the newest end |

**Response `200`**
```json
{
  "items": [
    {
      "timestamp": "2026-04-10T10:23:41.123456",
      "text": "तिमी मुर्ख हौ",
      "prediction": "OO",
      "confidence": 0.8732,
      "probabilities": {
        "NO": 0.08,
        "OO": 0.87,
        "OR": 0.03,
        "OS": 0.02
      },
      "preprocessed_text": "तिमी मुर्ख हौ",
      "emoji_features": { "total_emoji_count": 0, "..." : "..." }
    }
  ],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

**Pagination example:**
```
Page 1: GET /api/history?limit=20&offset=0
Page 2: GET /api/history?limit=20&offset=20
Page 3: GET /api/history?limit=20&offset=40
```

---

## 10. History — Stats

```
GET /api/history/stats
```

Returns aggregated statistics without fetching every record. Use for the dashboard summary row.

**Response `200` (with history)**
```json
{
  "total": 42,
  "avg_confidence": 0.8741,
  "class_counts": {
    "NO": 18,
    "OO": 12,
    "OR": 5,
    "OS": 7
  },
  "most_common_class": "NO"
}
```

**Response `200` (empty history)**
```json
{
  "total": 0,
  "avg_confidence": null,
  "class_counts": {},
  "most_common_class": null
}
```

---

## 11. History — Clear

```
DELETE /api/history
```

Permanently deletes the history file. No confirmation prompt — handle that in the UI.

**Response `200`**
```json
{
  "message": "History cleared. 42 record(s) deleted.",
  "deleted_count": 42
}
```

**Response `404`** (if already empty)
```json
{
  "detail": "History is already empty — nothing to clear."
}
```

---

## 12. Error Reference

All error responses follow FastAPI's standard shape:

```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning | When it happens |
|---|---|---|
| `422` | Validation error | Empty text, batch > 200, invalid field types |
| `503` | Service unavailable | Model still loading at startup, out of memory |
| `404` | Not found | History already empty on DELETE |
| `500` | Internal server error | Unexpected exception in inference or XAI |

---

## 13. TypeScript Types

Copy these into your React/Vite project:

```typescript
// ── Labels ──────────────────────────────────────────────────────────────────
export type Label = "NO" | "OO" | "OR" | "OS" | "Error";

export const LABEL_META = {
  NO: { text: "Non-Offensive",     color: "#28a745" },
  OO: { text: "Other-Offensive",   color: "#ffc107" },
  OR: { text: "Offensive-Racist",  color: "#dc3545" },
  OS: { text: "Offensive-Sexist",  color: "#6f42c1" },
  Error: { text: "Error",          color: "#6c757d" },
} as const;

// ── /health ──────────────────────────────────────────────────────────────────
export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
}

// ── /api/status ───────────────────────────────────────────────────────────────
export interface StatusResponse {
  model_loaded: boolean;
  device: string;
  preprocessor: boolean;
  lime: boolean;
  shap: boolean;
  captum: boolean;
}

// ── /api/predict ──────────────────────────────────────────────────────────────
export interface PredictRequest {
  text: string;
  save_to_history?: boolean;
}

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

export interface PredictResponse {
  prediction: Label;
  confidence: number;
  probabilities: Record<Label, number>;
  original_text: string;
  preprocessed_text: string;
  emoji_features: EmojiFeatures;
  script_info: ScriptInfo | null;
  error: string | null;
}

// ── /api/analyze ──────────────────────────────────────────────────────────────
export interface AnalyzeRequest {
  text: string;
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

// ── /api/explain/* ────────────────────────────────────────────────────────────
export interface ExplainRequest {
  text: string;
  num_samples?: number; // LIME only, default 200
  n_steps?: number;     // Captum only, default 50
}

export interface WordScore {
  word: string;
  score: number;
}

export interface ExplainResponse {
  method: "LIME" | "SHAP" | "Captum-IG";
  prediction: Label;
  confidence: number;
  word_scores: WordScore[];
  preprocessed_text: string;
  convergence_delta: number | null; // Captum only
  error: string | null;
}

// ── /api/batch ────────────────────────────────────────────────────────────────
export interface BatchRequest {
  texts: string[];
}

export interface BatchResult {
  text: string;          // truncated to 80 chars
  full_text: string;
  prediction: Label;
  confidence: number;
  preprocessed_text: string;
  error?: string;
}

export interface BatchProgressLine {
  index: number;
  total: number;
  result: BatchResult;
}

export interface BatchDoneLine {
  done: true;
  total: number;
}

export type BatchStreamLine = BatchProgressLine | BatchDoneLine;

// ── /api/history ──────────────────────────────────────────────────────────────
export interface HistoryItem {
  timestamp: string; // ISO 8601
  text: string;
  prediction: Label;
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
```

---

## 14. Frontend Integration Guide

### Recommended call order on app load

```
1. GET /health            → poll until model_loaded === true
2. GET /api/status        → store capabilities, show/hide XAI buttons
3. Ready to accept input
```

### Single prediction flow

```
user submits text
  → POST /api/predict
  → show prediction badge (color from LABEL_META)
  → show probability bar chart (4 bars)
  → show preprocessing details (script_info + emoji_features)
  → if emoji_features.total_emoji_count > 0, show emoji breakdown panel
```

### Explainability flow

```
user selects LIME / SHAP / Captum tab
  → check status.lime / status.shap / status.captum before enabling tab
  → POST /api/explain/lime  (or /shap or /captum)
  → render horizontal bar chart from word_scores
    - sort by abs(score) descending
    - positive score → green bar
    - negative score → red bar
  → for Captum: show convergence_delta warning if > 0.05
```

### Batch flow

```
user pastes texts or uploads CSV
  → POST /api/batch
  → read response as NDJSON stream (see streaming example in §8)
  → update progress bar: (index + 1) / total * 100
  → append each result to results table as it arrives
  → on { done: true }, finalize and enable download CSV
```

### History flow

```
on History tab open:
  → GET /api/history/stats   → show summary metrics
  → GET /api/history?limit=20&offset=0  → show table

pagination:
  → GET /api/history?limit=20&offset=N

clear button:
  → confirm in UI first
  → DELETE /api/history
```

### CORS

The backend allows requests from `http://localhost:5173` (Vite default) and `http://localhost:3000` (CRA default). If you deploy the frontend to a different URL, set the `FRONTEND_URL` environment variable before starting the server:

```bash
FRONTEND_URL=https://yourapp.vercel.app uvicorn backend.app.main:app ...
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/saved_models/xlm_roberta_results/large_final` | Local model path. Falls back to HuggingFace if not found |
| `HF_MODEL_ID` | `UDHOV/xlm-roberta-large-nepali-hate-classification` | HuggingFace model ID |
| `HISTORY_FILE` | `data/prediction_history.jsonl` | History file location |
| `FRONTEND_URL` | `""` | Additional CORS origin for deployed frontend |
