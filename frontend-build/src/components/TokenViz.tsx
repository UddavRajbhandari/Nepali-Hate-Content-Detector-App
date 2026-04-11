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
 * Color mapping:
 *   green  = positive = "for" the predicted class
 *   red    = negative = "against" the predicted class
 *   yellow = weakly positive (borderline)
 *   grey   = neutral
 */
function scoreToColor(score: number): { bg: string; border: string; color: string } {
  if (score > 0.6)  return { bg: "rgba(74,222,128,0.25)",  border: "#166534", color: "#4ade80" };  // strong for
  if (score > 0.3)  return { bg: "rgba(74,222,128,0.12)",  border: "#1a3d25", color: "#86efac" };  // medium for
  if (score > 0.05) return { bg: "rgba(251,191,36,0.12)",  border: "#4d3000", color: "#fbbf24" };  // weak for
  if (score < -0.3) return { bg: "rgba(248,113,113,0.25)", border: "#7f1d1d", color: "#fca5a5" };  // strong against
  if (score < -0.1) return { bg: "rgba(248,113,113,0.12)", border: "#5a1a1a", color: "#f87171" };  // medium against
  return { bg: "rgba(107,117,137,0.1)", border: "var(--border)", color: "var(--text2)" };          // neutral
}

export default function TokenViz({ wordScores, showTable = false }: Props) {
  const maxAbs = Math.max(...wordScores.map((w) => Math.abs(w.score)), 0.001);

  return (
    <div>
      {/* chip visualization */}
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
              title={`score: ${ws.score.toFixed(4)}`}
            >
              {ws.word}
            </span>
          );
        })}
      </div>

      {/* legend */}
      <div style={{ display: "flex", gap: 16, marginBottom: showTable ? 16 : 0 }}>
        {[
          { bg: "rgba(74,222,128,0.25)",  label: "For predicted class" },
          { bg: "rgba(248,113,113,0.25)", label: "Against predicted class" },
          { bg: "rgba(107,117,137,0.1)",  label: "Neutral" },
        ].map((l) => (
          <span key={l.label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--text3)" }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: l.bg, display: "inline-block" }} />
            {l.label}
          </span>
        ))}
      </div>

      {/* optional table */}
      {showTable && (
        <details style={{ marginTop: 4 }}>
          <summary>Score table</summary>
          <div style={{ marginTop: 10, maxHeight: 240, overflowY: "auto", borderRadius: 8, border: "1px solid var(--border)" }}>
            <table className="score-table">
              <thead>
                <tr>
                  <th>Token</th>
                  <th>Score</th>
                  <th>Direction</th>
                </tr>
              </thead>
              <tbody>
                {[...wordScores]
                  .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
                  .map((ws, i) => (
                    <tr key={i}>
                      <td className="mono">{ws.word}</td>
                      <td className="mono">{ws.score.toFixed(4)}</td>
                      <td>
                        <span style={{
                          color: ws.score > 0.05 ? "var(--green)" : ws.score < -0.05 ? "var(--red)" : "var(--text3)",
                          fontSize: 12
                        }}>
                          {ws.score > 0.05 ? "▲ for" : ws.score < -0.05 ? "▼ against" : "— neutral"}
                        </span>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}