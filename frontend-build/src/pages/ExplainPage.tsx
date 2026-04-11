import { useState, useEffect } from "react";
import { api, type ExplainResult, type StatusResponse } from "../utils/api";
import TokenViz from "../components/TokenViz";

interface Props {
  initialText: string;
  /** Capabilities from GET /api/status — controls which methods are enabled */
  caps: StatusResponse | null;
}

type Method = "lime" | "shap" | "captum";

const METHOD_INFO: Record<Method, {
  label: string;
  desc: string;
  slow: boolean;
  capsKey: keyof StatusResponse;
}> = {
  lime: {
    label: "LIME",
    desc: "Perturbs preprocessed tokens, fits a local linear model. Fast and reliable.",
    slow: false,
    capsKey: "lime",
  },
  shap: {
    label: "SHAP",
    desc: "Shapley values — theoretically grounded, slightly slower than LIME.",
    slow: false,
    capsKey: "shap",
  },
  captum: {
    label: "Captum (IG)",
    desc: "Integrated Gradients on subword tokens. Most precise but memory-intensive.",
    slow: true,
    capsKey: "captum",
  },
};

export default function ExplainPage({ initialText, caps }: Props) {
  const [text, setText]             = useState(initialText);
  const [method, setMethod]         = useState<Method>("lime");
  const [numSamples, setNumSamples] = useState(200); // matches API default
  const [nSteps, setNSteps]         = useState(50);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState<ExplainResult | null>(null);
  const [error, setError]           = useState<string | null>(null);

  // Sync when user types in Predict tab then switches here
  useEffect(() => {
    if (initialText) setText(initialText);
  }, [initialText]);

  // If the currently selected method becomes unavailable, fall back to lime
  useEffect(() => {
    if (caps && !caps[METHOD_INFO[method].capsKey]) {
      const fallback = (["lime", "shap", "captum"] as Method[]).find(
        (m) => caps[METHOD_INFO[m].capsKey]
      );
      if (fallback) setMethod(fallback);
    }
  }, [caps, method]);

  const isAvailable = (m: Method) =>
    caps === null || Boolean(caps[METHOD_INFO[m].capsKey]);

  const handleExplain = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r =
        method === "lime"   ? await api.explainLime(text, numSamples)
        : method === "shap" ? await api.explainShap(text)
        :                     await api.explainCaptum(text, nSteps);
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Explanation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ marginBottom: 6 }}>Explainability</h1>
        <p className="muted">Token-level attribution — what drove the model's decision.</p>
      </div>

      <div className="row">
        {/* ── Controls ── */}
        <div style={{ width: 300, flexShrink: 0 }}>
          <div className="card">
            <label htmlFor="ex-text">Text</label>
            <textarea id="ex-text" value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter Nepali text…"
              style={{ minHeight: 120, marginBottom: 16 }} />

            <h3 style={{ marginBottom: 10 }}>Method</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
              {(Object.keys(METHOD_INFO) as Method[]).map((m) => {
                const available = isAvailable(m);
                return (
                  <label key={m} style={{
                    display: "flex", alignItems: "flex-start", gap: 10,
                    padding: "10px 12px", borderRadius: 8,
                    cursor: available ? "pointer" : "not-allowed",
                    opacity: available ? 1 : 0.45,
                    marginBottom: 0,
                    border: `1px solid ${method === m ? "var(--accent2)" : "var(--border)"}`,
                    background: method === m ? "var(--accent-dim)" : "var(--bg3)",
                    transition: "all 0.15s",
                  }}>
                    <input type="radio" name="explain-method" value={m}
                      checked={method === m}
                      disabled={!available}
                      onChange={() => available && setMethod(m)}
                      style={{ marginTop: 2, accentColor: "var(--accent2)" }} />
                    <div>
                      <div style={{ fontSize: 13.5, fontWeight: 500, color: method === m ? "var(--accent)" : "var(--text)" }}>
                        {METHOD_INFO[m].label}
                        {METHOD_INFO[m].slow && (
                          <span className="tag" style={{ marginLeft: 6, fontSize: 10 }}>slow</span>
                        )}
                        {!available && (
                          <span className="tag" style={{ marginLeft: 6, fontSize: 10, borderColor: "var(--red)", color: "var(--red)" }}>
                            not installed
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 12, color: "var(--text3)", marginTop: 3, lineHeight: 1.4 }}>
                        {METHOD_INFO[m].desc}
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>

            {method === "lime" && isAvailable("lime") && (
              <div style={{ marginBottom: 16 }}>
                <label>Samples: {numSamples}</label>
                <input type="range" min={50} max={1000} step={50}
                  value={numSamples}
                  onChange={(e) => setNumSamples(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--accent2)" }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text3)" }}>
                  <span>50 (fast)</span><span>1000 (accurate)</span>
                </div>
              </div>
            )}

            {method === "captum" && isAvailable("captum") && (
              <div style={{ marginBottom: 16 }}>
                <label>Integration steps: {nSteps}</label>
                <input type="range" min={10} max={200} step={10}
                  value={nSteps}
                  onChange={(e) => setNSteps(Number(e.target.value))}
                  style={{ width: "100%", accentColor: "var(--accent2)" }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text3)", marginBottom: 8 }}>
                  <span>10</span><span>200</span>
                </div>
                <div className="alert alert-warn" style={{ fontSize: 12 }}>
                  Captum requires significant RAM. Use LIME or SHAP on cloud.
                </div>
              </div>
            )}

            <button className="btn btn-primary btn-full"
              onClick={handleExplain}
              disabled={loading || !text.trim() || !isAvailable(method)}>
              {loading
                ? <><span className="spinner" /> Explaining…</>
                : "◈ Generate explanation"}
            </button>
          </div>
        </div>

        {/* ── Result ── */}
        <div className="col">
          {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

          {!result && !loading && !error && (
            <div className="card empty-state">
              <div className="empty-state-icon">◈</div>
              Choose a method and generate an explanation
            </div>
          )}

          {loading && (
            <div className="card empty-state">
              <div style={{ marginBottom: 12 }}>
                <span className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }} />
              </div>
              <span className="muted">
                {method === "lime"
                  ? `Running LIME with ${numSamples} samples…`
                  : method === "shap"
                  ? "Computing Shapley values…"
                  : `Integrated Gradients (${nSteps} steps)…`}
              </span>
            </div>
          )}

          {result && (
            <>
              <div className="card" style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 12 }}>
                  <div>
                    <span className="tag" style={{ marginRight: 8 }}>{result.method}</span>
                    <span className={"pred-badge pred-" + result.prediction}>{result.prediction}</span>
                  </div>
                  <span className="confidence-pill">{(result.confidence * 100).toFixed(1)}% conf.</span>
                </div>

                <hr className="divider" />

                <label>Preprocessed text (what the model saw)</label>
                <div className="preprocess-box" style={{ marginBottom: 10 }}>
                  {result.preprocessed_text}
                </div>

                {/* Convergence delta — Captum only */}
                {result.convergence_delta !== null && result.convergence_delta !== undefined && (
                  <div style={{ fontSize: 12, color: "var(--text3)" }}>
                    Convergence Δ:{" "}
                    <span className="mono">{result.convergence_delta.toFixed(6)}</span>
                    {Math.abs(result.convergence_delta) < 0.05
                      ? <span style={{ color: "var(--green)", marginLeft: 6 }}>✓ good</span>
                      : <span style={{ color: "var(--amber)", marginLeft: 6 }}>⚠ increase steps</span>
                    }
                  </div>
                )}

                {/* SHAP fallback notice */}
                {result.error && (
                  <div className="alert alert-warn" style={{ marginTop: 10, fontSize: 12 }}>
                    Note: {result.error}
                  </div>
                )}
              </div>

              <div className="card">
                <h3 style={{ marginBottom: 14 }}>Token attribution</h3>
                {result.word_scores.length > 0
                  ? <TokenViz wordScores={result.word_scores} showTable />
                  : <p className="muted">No attribution scores were returned.</p>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
