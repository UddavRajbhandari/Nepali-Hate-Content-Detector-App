import { useState, useRef } from "react";
import { api, type PredictResult, type AnalyzeResponse } from "../utils/api";
import { LABEL_META, type Label } from "../utils/labels";
import ProbabilityBars from "../components/ProbabilityBars";

interface Props {
  onTextChange: (text: string) => void;
}

const EXAMPLES = [
  "यो राम्रो छ, मलाई मन पर्यो!",
  "timi ekdam murkha chau",
  "मुस्लिम हरु सबै खराब छन्, यो देशबाट जान पर्छ",
  "केटीहरु घरमा बस्नु पर्छ, काम गर्न जान हुँदैन",
  "नमस्ते! आज मौसम राम्रो छ 😊",
];

const SCRIPT_LABEL: Record<string, string> = {
  devanagari:     "Devanagari",
  romanized_nepali: "Romanized Nepali",
  english:        "English",
  mixed:          "Mixed script",
  other:          "Other",
};

export default function SinglePredict({ onTextChange }: Props) {
  const [text, setText]           = useState("");
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState<PredictResult | null>(null);
  const [analyze, setAnalyze]     = useState<AnalyzeResponse | null>(null);
  const [error, setError]         = useState<string | null>(null);
  const [save, setSave]           = useState(true);
  const textareaRef               = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setAnalyze(null);
    onTextChange(text);
    try {
      // Fire predict and analyze in parallel — analyze is lightweight
      const [r, a] = await Promise.all([
        api.predict(text, save),
        api.analyze(text),
      ]);
      setResult(r);
      setAnalyze(a);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const meta = result ? LABEL_META[result.prediction as Label] : null;
  const emojiCount = result?.emoji_features?.total_emoji_count ?? 0;

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ marginBottom: 6 }}>Single prediction</h1>
        <p className="muted">Enter Nepali text in Devanagari, Romanized, English, or mixed scripts.</p>
      </div>

      <div className="row">
        {/* ── Input ── */}
        <div className="col" style={{ maxWidth: 560 }}>
          <div className="card">
            <label htmlFor="text-input">Text input</label>
            <textarea
              id="text-input"
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={"यहाँ आफ्नो पाठ लेख्नुहोस्…\nOr: timi ramro chau\nOr: This is a test 😡"}
              style={{ marginBottom: 12, minHeight: 160 }}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(); }}
            />

            <div style={{ marginBottom: 14 }}>
              <span className="muted" style={{ fontSize: 12, marginRight: 8 }}>Examples:</span>
              {EXAMPLES.map((ex, i) => (
                <button key={i} className="btn btn-ghost btn-sm"
                  style={{ marginRight: 6, marginBottom: 6, fontSize: 12 }}
                  onClick={() => { setText(ex); textareaRef.current?.focus(); }}>
                  {i + 1}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <button className="btn btn-primary" onClick={handleSubmit}
                disabled={loading || !text.trim()}>
                {loading ? <span className="spinner" /> : "⬡"}
                {loading ? "Analyzing…" : "Analyze"}
              </button>
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "var(--text2)", cursor: "pointer", marginBottom: 0 }}>
                <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)}
                  style={{ width: 14, height: 14, accentColor: "var(--accent2)" }} />
                Save to history
              </label>
              {text && (
                <span className="muted" style={{ fontSize: 12, marginLeft: "auto" }}>
                  {text.length} chars · ⌘↵ to submit
                </span>
              )}
            </div>
          </div>

          {/* Quick-info card */}
          <div className="card" style={{ marginTop: 16 }}>
            <h3 style={{ marginBottom: 12 }}>Supported inputs</h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {[
                ["Devanagari",  "नेपाली पाठ"],
                ["Romanized",   "ma nepali xu"],
                ["English",     "Translated automatically"],
                ["Mixed script","yo ramro cha 😡"],
                ["Emojis",      "180+ mapped to Nepali"],
                ["Code-mixed",  "Nepali + English"],
              ].map(([k, v]) => (
                <div key={k} style={{ fontSize: 12.5 }}>
                  <span style={{ color: "var(--accent)", fontWeight: 500 }}>{k}</span>
                  <span className="muted"> — {v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Result ── */}
        <div className="col">
          {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

          {!result && !error && !loading && (
            <div className="card empty-state">
              <div className="empty-state-icon">⬡</div>
              Results will appear here
            </div>
          )}

          {loading && (
            <div className="card empty-state">
              <div style={{ marginBottom: 12 }}>
                <span className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }} />
              </div>
              <span className="muted">Preprocessing &amp; inferring…</span>
            </div>
          )}

          {result && meta && (
            <>
              {/* Prediction card */}
              <div className="card" style={{
                borderColor: meta.border,
                background: `linear-gradient(135deg, var(--bg2) 55%, ${meta.bg})`,
              }}>
                <div className="result-header">
                  <div>
                    <div style={{ marginBottom: 8 }}>
                      <span className={"pred-badge pred-" + result.prediction}>{result.prediction}</span>
                    </div>
                    <div className="result-label" style={{ color: meta.color }}>{meta.title}</div>
                    <p className="muted" style={{ marginTop: 6, fontSize: 13.5 }}>{meta.desc}</p>
                  </div>
                  <span className="confidence-pill">{(result.confidence * 100).toFixed(1)}% conf.</span>
                </div>
                <hr className="divider" />
                <h3 style={{ marginBottom: 12 }}>Class probabilities</h3>
                <ProbabilityBars probabilities={result.probabilities} />
              </div>

              {/* Preprocessing card */}
              <div className="card" style={{ marginTop: 16 }}>
                <h3 style={{ marginBottom: 12 }}>Preprocessing</h3>
                <div className="row" style={{ gap: 12 }}>
                  <div className="col">
                    <label>Original</label>
                    <div className="preprocess-box">{result.original_text}</div>
                  </div>
                  <div className="col">
                    <label>Preprocessed</label>
                    <div className="preprocess-box">{result.preprocessed_text || "—"}</div>
                  </div>
                </div>

                {/* Script info row — from /api/analyze response */}
                {analyze && (
                  <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
                    <span className="tag">
                      {SCRIPT_LABEL[analyze.script_info.script_type] ?? analyze.script_info.script_type}
                    </span>
                    <span className="tag">
                      script confidence {(analyze.script_info.confidence * 100).toFixed(0)}%
                    </span>
                    {analyze.emoji_info.total_count > 0 && (
                      <span className="tag">
                        {analyze.emoji_info.total_count} emoji · {analyze.emoji_info.known_count} mapped
                      </span>
                    )}
                  </div>
                )}

                {/* Emoji breakdown — only shown when emojis found */}
                {emojiCount > 0 && (
                  <div style={{ marginTop: 12, display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {([
                      ["Total",    result.emoji_features.total_emoji_count],
                      ["Hate",     result.emoji_features.hate_emoji_count],
                      ["Positive", result.emoji_features.positive_emoji_count],
                      ["Mockery",  result.emoji_features.mockery_emoji_count],
                      ["Sadness",  result.emoji_features.sadness_emoji_count],
                    ] as [string, number][]).map(([k, v]) =>
                      v > 0 ? (
                        <span key={k} className="tag">{k}: {v}</span>
                      ) : null
                    )}
                    {result.emoji_features.has_mixed_sentiment === 1 && (
                      <span className="tag" style={{ borderColor: "var(--amber)", color: "var(--amber)" }}>
                        mixed sentiment
                      </span>
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
