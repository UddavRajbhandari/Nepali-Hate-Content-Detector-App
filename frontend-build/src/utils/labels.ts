export type Label = "NO" | "OO" | "OR" | "OS";

export const LABEL_META: Record<
  Label,
  { color: string; bg: string; border: string; title: string; desc: string }
> = {
  NO: {
    color: "#4ade80",
    bg: "rgba(74,222,128,0.08)",
    border: "#166534",
    title: "Non-Offensive",
    desc: "The text does not contain hate speech or offensive content.",
  },
  OO: {
    color: "#fbbf24",
    bg: "rgba(251,191,36,0.08)",
    border: "#78350f",
    title: "Other-Offensive",
    desc: "Contains general offensive language not targeting a specific group.",
  },
  OR: {
    color: "#f87171",
    bg: "rgba(248,113,113,0.08)",
    border: "#7f1d1d",
    title: "Offensive-Racist",
    desc: "Contains hate speech targeting race, ethnicity, caste, or religion.",
  },
  OS: {
    color: "#c084fc",
    bg: "rgba(192,132,252,0.08)",
    border: "#4c1d95",
    title: "Offensive-Sexist",
    desc: "Contains hate speech targeting gender or sexual orientation.",
  },
};

export const CHART_COLORS: Record<Label, string> = {
  NO: "#4ade80",
  OO: "#fbbf24",
  OR: "#f87171",
  OS: "#c084fc",
};
