import { CHART_COLORS, type Label } from "../utils/labels";

interface Props {
  probabilities: Record<string, number>;
}

const BAR_BG: Record<string, string> = {
  NO: "rgba(74,222,128,0.15)",
  OO: "rgba(251,191,36,0.15)",
  OR: "rgba(248,113,113,0.15)",
  OS: "rgba(192,132,252,0.15)",
};

export default function ProbabilityBars({ probabilities }: Props) {
  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a);
  return (
    <div className="prob-grid">
      {sorted.map(([label, prob]) => (
        <div key={label} className="prob-row">
          <span className="prob-label" style={{ color: CHART_COLORS[label as Label] }}>
            {label}
          </span>
          <div className="prob-bar-track" style={{ background: BAR_BG[label] ?? "var(--bg4)" }}>
            <div
              className="prob-bar-fill"
              style={{
                width: `${(prob * 100).toFixed(1)}%`,
                background: CHART_COLORS[label as Label] ?? "#6b7589",
              }}
            />
          </div>
          <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
