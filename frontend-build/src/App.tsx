import { useState, useEffect, useRef } from "react";
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
  const [tab, setTab]               = useState<Tab>("predict");
  const [lastText, setLastText]     = useState("");
  const [boot, setBoot]             = useState<BootState>("polling");
  const [caps, setCaps]             = useState<StatusResponse | null>(null);
  const [device, setDevice]         = useState("");
  // Banner shown when the server drops AFTER initial boot
  const [disconnected, setDisconnected] = useState(false);
  const keepaliveRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Initial boot: poll /health until model_loaded ──────────────────────────
  // bootReady ref avoids stale closure bug: setTimeout captures the initial
  // value of `boot` ("polling") and never sees the updated "ready" value.
  // Using a ref lets the timeout read the live value without re-rendering.
  const bootReadyRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      while (!cancelled) {
        try {
          const h = await api.health();
          if (h.model_loaded) {
            setDevice(h.device);
            const s = await api.status();
            if (!cancelled) {
              setCaps(s);
              bootReadyRef.current = true;  // mark ready BEFORE setState
              setBoot("ready");
            }
            return;
          }
        } catch {
          // Server not up yet — keep polling silently
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };

    // 120 s timeout — reads ref, not stale closure variable
    const timeout = setTimeout(() => {
      if (!bootReadyRef.current) setBoot("error");
    }, 300_000 );

    poll().catch(() => { if (!cancelled) setBoot("error"); });

    return () => {
      cancelled = true;
      clearTimeout(timeout);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Keepalive: check every 30 s after boot so we detect server restarts ────
  useEffect(() => {
    if (boot !== "ready") return;

    keepaliveRef.current = setInterval(async () => {
      try {
        const h = await api.health();
        if (h.model_loaded) {
          // Server is back or still up — hide banner
          setDisconnected(false);
        } else {
          // Server up but model still loading (e.g. after restart)
          setDisconnected(true);
        }
      } catch {
        // Server unreachable — show reconnecting banner
        setDisconnected(true);
      }
    }, 30_000);

    return () => {
      if (keepaliveRef.current) clearInterval(keepaliveRef.current);
    };
  }, [boot]);

  // ── Boot screen ─────────────────────────────────────────────────────────────
  if (boot === "polling") {
    return (
      <div className="app">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", gap: 16 }}>
          <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
          <p className="muted">Loading model…</p>
          <p style={{ fontSize: 12, color: "var(--text3)" }}>
            This may take 60–90 seconds on first start while the model downloads.
          </p>
        </div>
      </div>
    );
  }

  if (boot === "error") {
    return (
      <div className="app">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", gap: 12 }}>
          <div className="alert alert-error" style={{ maxWidth: 420, textAlign: "center" }}>
            Could not connect to the backend after 5 minutes.
            <br />The model server may still be loading or the HF Space may be sleeping.
            <br /><span style={{ fontSize: 12, opacity: 0.8 }}>Click Retry to try again.</span>
          </div>
          <button className="btn btn-ghost" onClick={() => setBoot("polling")}>
            ↺ Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* ── Disconnected banner — shown without blocking the UI ── */}
      {disconnected && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, zIndex: 200,
          background: "var(--amber-bg)", borderBottom: "1px solid var(--amber)",
          padding: "8px 16px", textAlign: "center",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
          fontSize: 13, color: "var(--amber)",
        }}>
          <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2, borderTopColor: "var(--amber)" }} />
          Backend unreachable — reconnecting… Start the server if it is not running.
          <button
            className="btn btn-ghost btn-sm"
            style={{ fontSize: 12, padding: "2px 8px", borderColor: "var(--amber)", color: "var(--amber)" }}
            onClick={async () => {
              try {
                const h = await api.health();
                if (h.model_loaded) setDisconnected(false);
              } catch { /* still down */ }
            }}
          >
            Check now
          </button>
        </div>
      )}

      <header className="header" style={disconnected ? { marginTop: 37 } : {}}>
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