import { useState, useEffect, useCallback } from "react";
import { api, type HistoryItem, type HistoryStatsResponse } from "../utils/api";
import { CHART_COLORS, type Label } from "../utils/labels";
import {
  BarChart, Bar, Cell,
  XAxis, YAxis, Tooltip,
  LineChart, Line, CartesianGrid,
  ResponsiveContainer,
} from "recharts";

const PAGE_SIZE = 20;

export default function HistoryPage() {
  const [items, setItems]         = useState<HistoryItem[]>([]);
  const [stats, setStats]         = useState<HistoryStatsResponse | null>(null);
  const [offset, setOffset]       = useState(0);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState<string | null>(null);
  const [clearing, setClearing]   = useState(false);

  // Fetch stats + first page in parallel
  const load = useCallback(async (newOffset = 0) => {
    setLoading(true);
    setError(null);
    try {
      const [histRes, statsRes] = await Promise.all([
        api.getHistory(PAGE_SIZE, newOffset),
        api.getHistoryStats(),
      ]);
      setItems(histRes.items); // backend already returns newest-first
      setStats(statsRes);
      setOffset(newOffset);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load history");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(0); }, [load]);

  const handleClear = async () => {
    if (!confirm("Clear all prediction history? This cannot be undone.")) return;
    setClearing(true);
    try {
      await api.clearHistory();
      setItems([]);
      setStats(null);
      setOffset(0);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "";
      if (!msg.toLowerCase().includes("already empty")) {
        setError(msg || "Failed to clear history");
      } else {
        setItems([]);
        setStats(null);
        setOffset(0);
      }
    } finally {
      setClearing(false);
    }
  };

  // Chart data
  const barData = stats
    ? Object.entries(stats.class_counts).map(([label, count]) => ({ label, count }))
    : [];

  const byDate = items.reduce<Record<string, number>>((acc, it) => {
    const d = it.timestamp.slice(0, 10);
    acc[d] = (acc[d] ?? 0) + 1;
    return acc;
  }, {});
  const lineData = Object.entries(byDate)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, count]) => ({ date: date.slice(5), count }));

  const hasHistory = stats !== null && stats.total > 0;
  const totalPages = stats ? Math.ceil(stats.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
        <div>
          <h1 style={{ marginBottom: 6 }}>Prediction history</h1>
          <p className="muted">All saved predictions from the Single Predict tab.</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-ghost btn-sm" onClick={() => load(offset)} disabled={loading}>
            ↺ Refresh
          </button>
          {hasHistory && (
            <button className="btn btn-danger btn-sm" onClick={handleClear} disabled={clearing}>
              {clearing ? "Clearing…" : "🗑 Clear all"}
            </button>
          )}
        </div>
      </div>

      {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

      {loading && (
        <div className="card empty-state">
          <span className="spinner" style={{ width: 24, height: 24, borderWidth: 3 }} />
        </div>
      )}

      {!loading && !hasHistory && (
        <div className="card empty-state">
          <div className="empty-state-icon">◷</div>
          No history yet. Make predictions with "Save to history" enabled.
        </div>
      )}

      {!loading && hasHistory && stats && (
        <>
          {/* Stats row */}
          <div className="hist-grid" style={{ marginBottom: 20 }}>
            {[
              { label: "Total predictions",  value: stats.total },
              { label: "Avg confidence",     value: stats.avg_confidence !== null ? `${(stats.avg_confidence * 100).toFixed(1)}%` : "—" },
              { label: "Most common class",  value: stats.most_common_class ?? "—" },
              { label: "Unique classes",     value: Object.keys(stats.class_counts).length },
            ].map((s) => (
              <div key={s.label} className="hist-stat">
                <div className="hist-stat-val">{s.value}</div>
                <div className="hist-stat-label">{s.label}</div>
              </div>
            ))}
          </div>

          {/* Charts */}
          <div className="row" style={{ marginBottom: 20 }}>
            <div className="card col">
              <h3 style={{ marginBottom: 14 }}>Class distribution</h3>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={barData} barCategoryGap="30%">
                  <XAxis dataKey="label" tick={{ fill: "var(--text2)", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "var(--text3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 13 }} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {barData.map((entry) => (
                      <Cell key={entry.label} fill={CHART_COLORS[entry.label as Label] ?? "#6b7589"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {lineData.length > 1 && (
              <div className="card col">
                <h3 style={{ marginBottom: 14 }}>Predictions over time</h3>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={lineData}>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="date" tick={{ fill: "var(--text3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "var(--text3)", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 13 }} />
                    <Line type="monotone" dataKey="count" stroke="var(--accent2)" strokeWidth={2} dot={{ r: 3, fill: "var(--accent2)" }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Table */}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3>Recent predictions</h3>
              <span className="muted" style={{ fontSize: 12 }}>
                Page {currentPage} of {totalPages} · {stats.total} total
              </span>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="batch-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Text</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((it, i) => (
                    <tr key={i}>
                      <td className="mono-sm" style={{ color: "var(--text3)", whiteSpace: "nowrap" }}>
                        {new Date(it.timestamp).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      </td>
                      <td style={{ maxWidth: 380 }}>
                        <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={it.text}>
                          {it.text}
                        </div>
                      </td>
                      <td><span className={"pred-badge pred-" + it.prediction}>{it.prediction}</span></td>
                      <td className="mono-sm">{(it.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div style={{ padding: "12px 16px", borderTop: "1px solid var(--border)", display: "flex", gap: 8, justifyContent: "center" }}>
                <button className="btn btn-ghost btn-sm"
                  disabled={offset === 0 || loading}
                  onClick={() => load(offset - PAGE_SIZE)}>
                  ← Prev
                </button>
                <span className="muted" style={{ fontSize: 13, lineHeight: "30px" }}>
                  {currentPage} / {totalPages}
                </span>
                <button className="btn btn-ghost btn-sm"
                  disabled={offset + PAGE_SIZE >= (stats?.total ?? 0) || loading}
                  onClick={() => load(offset + PAGE_SIZE)}>
                  Next →
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
