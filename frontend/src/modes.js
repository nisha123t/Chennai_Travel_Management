// Colours follow each network's own identity: CMRL blue, MTC red, suburban ochre.
export const MODES = {
  metro: { label: "Metro", color: "#1f5fad" },
  rail: { label: "Suburban Rail", color: "#b7790f" },
  bus: { label: "MTC Bus", color: "#c2362b" },
  walk: { label: "Walk", color: "#7c8b89" },
  auto: { label: "Auto", color: "#4f7d2a" },
  cab: { label: "Cab", color: "#6b4fa0" },
};

export const mode = (m) => MODES[m] || { label: m, color: "#555" };

export const fmtMin = (m) => {
  const v = Math.round(m);
  if (v < 60) return `${v} min`;
  return `${Math.floor(v / 60)} h ${String(v % 60).padStart(2, "0")} min`;
};

export const fmtHour = (h) => {
  const suffix = h < 12 ? "am" : "pm";
  const hh = h % 12 === 0 ? 12 : h % 12;
  return `${hh}:00 ${suffix}`;
};
