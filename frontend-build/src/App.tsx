import { useState, useEffect } from "react";
import { api, type StatusResponse } from "./utils/api";
import SinglePredict from "./pages/SinglePredict";
import BatchAnalysis from "./pages/BatchAnalysis";
import ExplainPage from "./pages/ExplainPage";
import HistoryPage from "./pages/HistoryPage";
import "./app.css";

type Tab = "predict" | "explain" | "batch" | "history";

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "predict", label: "Predict",        icon: "⬡" },
  { id: "explain", label: "Explainability", icon: "◈" },
  { id: "batch",   label: "Batch",          icon: "▦" },
  { id: "history", label: "History",        icon: "◷" },
];

type BootState = "polling" | "ready" | "error";

export default function App() {
  const [tab, setTab]           = useState<Tab>("predict");
  const [lastText, setLastText] = useState("");
  const [boot, setBoot]         = useState<BootState>("polling");
  const [caps, setCaps]         = useState<StatusResponse | null>(null);
  const [device, setDevice]     = useState("");

  // ── On mount: poll /health until model_loaded, then fetch /api/status ──
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      while (!cancelled) {
        try {
          const h = await api.health();
          if (h.model_loaded) {
            setDevice(h.device);
            // Now fetch capabilities so XAI buttons know what to show
            const s = await api.status();
            if (!cancelled) {
              setCaps(s);
              setBoot("ready");
            }
            return;
          }
        } catch {
          // Server not up yet — keep polling
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };

    // 30 s hard timeout — avoids infinite spin if something is wrong
    const timeout = setTimeout(() => {
      if (boot !== "ready") setBoot("error");
    }, 30_000);

    poll().catch(() => { if (!cancelled) setBoot("error"); });

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Boot screen ──────────────────────────────────────────────────────────
  if (boot === "polling") {
    return (
      <div className="app">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", gap: 16 }}>
          <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
          <p className="muted">Loading model…</p>
          <p style={{ fontSize: 12, color: "var(--text3)" }}>This may take up to 30 seconds on first start.</p>
        </div>
      </div>
    );
  }

  if (boot === "error") {
    return (
      <div className="app">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", gap: 12 }}>
          <div className="alert alert-error" style={{ maxWidth: 420, textAlign: "center" }}>
            Could not connect to the backend after 30 seconds.
            <br />Check that the server is running on port 8000.
          </div>
          <button className="btn btn-ghost" onClick={() => { setBoot("polling"); }}>
            ↺ Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="wordmark">
            <span className="wordmark-icon">🛡</span>
            <div>
              <div className="wordmark-title">Nepali Hate Detector</div>
              <div className="wordmark-sub">
                XLM-RoBERTa Large · 4-class · {device}
              </div>
            </div>
          </div>
          <nav className="tabs">
            {TABS.map((t) => (
              <button
                key={t.id}
                className={"tab-btn" + (tab === t.id ? " active" : "")}
                onClick={() => setTab(t.id)}
              >
                <span className="tab-icon">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        {tab === "predict" && <SinglePredict onTextChange={setLastText} />}
        {tab === "explain" && <ExplainPage initialText={lastText} caps={caps} />}
        {tab === "batch"   && <BatchAnalysis />}
        {tab === "history" && <HistoryPage />}
      </main>
    </div>
  );
}
