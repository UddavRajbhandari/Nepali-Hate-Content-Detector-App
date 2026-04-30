import type { WordScore } from "../utils/api";

interface Props {
  wordScores: WordScore[];
  showTable?: boolean;
}

/**
 * Score semantics (consistent across LIME, SHAP, Captum):
 *   positive score → word pushes prediction TOWARD the predicted class
 *   negative score → word pushes prediction AWAY from the predicted class
 *
 * All thresholds are RELATIVE to the max absolute score in the result set.
 * This is critical: a non-offensive text has tiny raw scores (e.g. 0.016 max)
 * so a fixed threshold of 0.05 would mark everything neutral. By normalising
 * first (score / maxAbs), the most influential word always scores ±1.0 and
 * thresholds are meaningful regardless of absolute scale.
 *
 * Neutral = word contributed less than 20% of the strongest signal.
 */
function scoreToColor(norm: number): { bg: string; border: string; color: string } {
  if (norm > 0.6)  return { bg: "rgba(74,222,128,0.25)",  border: "#166534", color: "#4ade80" };
  if (norm > 0.3)  return { bg: "rgba(74,222,128,0.12)",  border: "#1a3d25", color: "#86efac" };
  if (norm > 0.2)  return { bg: "rgba(251,191,36,0.12)",  border: "#4d3000", color: "#fbbf24" };
  if (norm < -0.6) return { bg: "rgba(248,113,113,0.25)", border: "#7f1d1d", color: "#fca5a5" };
  if (norm < -0.3) return { bg: "rgba(248,113,113,0.12)", border: "#5a1a1a", color: "#f87171" };
  if (norm < -0.2) return { bg: "rgba(248,113,113,0.08)", border: "#3d1010", color: "#fca5a5" };
  return { bg: "rgba(107,117,137,0.1)", border: "var(--border)", color: "var(--text2)" };
}

function directionLabel(norm: number): { text: string; color: string } {
  if (norm > 0.2)  return { text: "▲ for",     color: "var(--green)" };
  if (norm < -0.2) return { text: "▼ against", color: "var(--red)" };
  return              { text: "— neutral",  color: "var(--text3)" };
}

export default function TokenViz({ wordScores, showTable = false }: Props) {
  const maxAbs = Math.max(...wordScores.map((w) => Math.abs(w.score)), 0.001);

  return (
    <div>
      <div className="token-viz" style={{ marginBottom: 20 }}>
        {wordScores.map((ws, i) => {
          const norm = ws.score / maxAbs;
          const { bg, border, color } = scoreToColor(norm);
          const opacity = 0.4 + 0.6 * Math.abs(norm);
          return (
            <span
              key={i}
              className="token-chip"
              style={{ background: bg, borderColor: border, color, opacity }}
              title={`raw: ${ws.score.toFixed(4)} | norm: ${norm.toFixed(2)}`}
            >
              {ws.word}
            </span>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 16, marginBottom: showTable ? 16 : 0, flexWrap: "wrap" }}>
        {[
          { bg: "rgba(74,222,128,0.25)",  label: "For predicted class" },
          { bg: "rgba(248,113,113,0.25)", label: "Against predicted class" },
          { bg: "rgba(107,117,137,0.1)",  label: "Neutral (< 20% of strongest signal)" },
        ].map((l) => (
          <span key={l.label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text3)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: l.bg, display: "inline-block" }} />
            {l.label}
          </span>
        ))}
      </div>

      {showTable && (
        <details style={{ marginTop: 4 }}>
          <summary>Score table</summary>
          <div style={{ marginTop: 10, maxHeight: 240, overflowY: "auto", borderRadius: 8, border: "1px solid var(--border)" }}>
            <table className="score-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>Raw score</th>
                  <th>Direction</th>
                </tr>
              </thead>
              <tbody>
                {[...wordScores]
                  .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
                  .map((ws, i) => {
                    const norm = ws.score / maxAbs;
                    const dir = directionLabel(norm);
                    return (
                      <tr key={i}>
                        <td className="mono">{ws.word}</td>
                        <td className="mono">{ws.score.toFixed(4)}</td>
                        <td>
                          <span style={{ color: dir.color, fontSize: 12 }}>
                            {dir.text}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}