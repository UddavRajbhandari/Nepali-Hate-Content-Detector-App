import { useState, useRef } from "react";
import { flushSync } from "react-dom";
import { api, type BatchResultItem } from "../utils/api";
import { CHART_COLORS, type Label } from "../utils/labels";

export default function BatchAnalysis() {
  const [inputMode, setInputMode] = useState<"text" | "csv">("text");
  const [textInput, setTextInput] = useState("");
  const [results, setResults] = useState<BatchResultItem[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Ref for the file input so we can reset it after upload
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleRun = async () => {
    // Always read texts from textInput regardless of inputMode —
    // CSV upload populates textInput and switches mode to "text"
    const texts = textInput.split("\n").map((l) => l.trim()).filter(Boolean);

    if (!texts.length) {
      setError("No texts to analyze. Enter at least one line of text.");
      return;
    }
    if (texts.length > 200) {
      setError(`Too many texts (${texts.length}). Maximum is 200 per batch.`);
      return;
    }

    setLoading(true);
    setError(null);
    setResults([]);
    setProgress({ done: 0, total: texts.length });
    abortRef.current = new AbortController();

    try {
      const acc: BatchResultItem[] = [];
      await api.batchStream(
        texts,
        (item, index, total) => {
          acc.push(item);
          // flushSync forces React to render immediately after each streamed item
          // so the progress bar and results table update in real time rather than
          // all at once at the end (React 18 batches async state updates by default)
          flushSync(() => {
            setResults([...acc]);
            setProgress({ done: index + 1, total });
          });
        },
        (total) => {
          flushSync(() => setProgress({ done: total, total }));
        },
        abortRef.current.signal
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError(e instanceof Error ? e.message : "Batch failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCSV = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset the file input immediately so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = "";

    const { default: Papa } = await import("papaparse");
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const cols = res.meta.fields ?? [];
        if (!cols.length) {
          setError("CSV appears to have no columns.");
          return;
        }
        // Prefer a column whose name contains "text", otherwise use the first column
        const textCol = cols.find((c) => c.toLowerCase().includes("text")) ?? cols[0];
        const texts = res.data
          .map((row) => row[textCol]?.trim())
          .filter((t): t is string => Boolean(t));

        if (!texts.length) {
          setError(`No text values found in column "${textCol}".`);
          return;
        }

        // Populate text area and switch mode — user can review before running
        setTextInput(texts.join("\n"));
        setInputMode("text");
        setError(null);
      },
      error: (err) => setError(`CSV parse error: ${err.message}`),
    });
  };

  const downloadCSV = () => {
    const header = "text,prediction,confidence,preprocessed_text\n";
    const rows = results
      .map((r) =>
        [
          `"${r.full_text.replace(/"/g, '""')}"`,
          r.prediction,
          r.confidence.toFixed(4),
          `"${r.preprocessed_text.replace(/"/g, '""')}"`,
        ].join(",")
      )
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `batch_results_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const summary = results.reduce<Record<string, number>>((acc, r) => {
    acc[r.prediction] = (acc[r.prediction] ?? 0) + 1;
    return acc;
  }, {});

  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0;
  const textCount = textInput.split("\n").filter((l) => l.trim()).length;

  return (
    <div>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ marginBottom: 6 }}>Batch analysis</h1>
        <p className="muted">Analyze multiple texts at once with live streaming progress.</p>
      </div>

      {/* ── Input card ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          {(["text", "csv"] as const).map((m) => (
            <button
              key={m}
              className="btn btn-ghost btn-sm"
              style={inputMode === m ? { borderColor: "var(--accent2)", color: "var(--accent)" } : {}}
              onClick={() => setInputMode(m)}
            >
              {m === "text" ? "▦ Text area" : "📄 CSV upload"}
            </button>
          ))}
        </div>

        {inputMode === "text" ? (
          <>
            <label>One text per line (max 200)</label>
            <textarea
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder={"यो राम्रो छ\ntimi murkha chau\nThis is a test"}
              style={{ minHeight: 200, fontFamily: "monospace", fontSize: 13 }}
            />
            <div style={{ fontSize: 12, color: "var(--text3)", marginTop: 6 }}>
              {textCount} {textCount === 1 ? "text" : "texts"} ready
              {textCount > 200 && (
                <span style={{ color: "var(--red)", marginLeft: 8 }}>
                  ⚠ exceeds 200 limit
                </span>
              )}
            </div>
          </>
        ) : (
          <div style={{ textAlign: "center", padding: "32px 0" }}>
            <label style={{ fontSize: 14, color: "var(--text2)", cursor: "pointer" }}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleCSV}
                style={{ display: "none" }}
              />
              <span style={{
                display: "inline-block", padding: "10px 24px",
                border: "1px dashed var(--border2)", borderRadius: 10,
                background: "var(--bg3)", cursor: "pointer",
                transition: "border-color 0.15s",
              }}>
                Click to upload CSV
              </span>
            </label>
            <p className="muted" style={{ marginTop: 10, fontSize: 13 }}>
              Must contain a column named "text" (or similar).
              After upload, texts load into the text area for review before running.
            </p>
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
          <button
            className="btn btn-primary"
            onClick={handleRun}
            disabled={loading || textCount === 0 || textCount > 200}
          >
            {loading ? <><span className="spinner" /> Analyzing…</> : "▦ Run batch"}
          </button>

          {loading && (
            <button className="btn btn-ghost" onClick={() => abortRef.current?.abort()}>
              ✕ Cancel
            </button>
          )}

          {results.length > 0 && !loading && (
            <button
              className="btn btn-ghost"
              onClick={downloadCSV}
              style={{ marginLeft: "auto" }}
            >
              ↓ Download CSV
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>
      )}

      {/* ── Progress bar ── */}
      {loading && progress && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 8 }}>
            <span className="muted">Processing…</span>
            <span className="mono-sm">{progress.done} / {progress.total} ({pct}%)</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}

      {/* ── Summary strip ── */}
      {results.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--text3)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Completed
              </div>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: 26, letterSpacing: "-0.04em", color: "var(--accent)" }}>
                {results.length}
              </div>
            </div>

            {Object.entries(summary).map(([label, count]) => (
              <div
                key={label}
                style={{ borderLeft: `3px solid ${CHART_COLORS[label as Label] ?? "#888"}`, paddingLeft: 12 }}
              >
                <div style={{ fontSize: 11, color: "var(--text3)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {label}
                </div>
                <div style={{ fontSize: 20, fontFamily: "'Fraunces', serif", color: CHART_COLORS[label as Label] ?? "var(--text)" }}>
                  {count}{" "}
                  <span style={{ fontSize: 13, color: "var(--text3)", fontFamily: "inherit" }}>
                    ({((count / results.length) * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Results table ── */}
      {results.length > 0 && (
        <div className="card" style={{ padding: 0, overflow: "hidden" }}>
          <div style={{ maxHeight: 520, overflowY: "auto" }}>
            <table className="batch-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Text</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td className="mono-sm" style={{ color: "var(--text3)" }}>{i + 1}</td>
                    <td style={{ maxWidth: 340 }}>
                      <div
                        style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                        title={r.full_text}
                      >
                        {r.text}
                      </div>
                      {r.error && (
                        <div style={{ fontSize: 11, color: "var(--red)", marginTop: 2 }}>{r.error}</div>
                      )}
                    </td>
                    <td>
                      <span className={"pred-badge pred-" + r.prediction}>{r.prediction}</span>
                    </td>
                    <td className="mono-sm">{(r.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && results.length === 0 && !error && (
        <div className="card empty-state">
          <div className="empty-state-icon">▦</div>
          Enter texts above and click Run batch
        </div>
      )}
    </div>
  );
}