import type { Data, Layout } from "plotly.js-dist-min";
import type { Epoch } from "./api";

export const MAX_COMPARE = 8; // one fixed categorical color per run; never cycle past the palette

/** Base metric names in a history ("loss", "rmse", …); validation variants are the same names prefixed "val_". */
export function metricNames(history: Epoch[]): string[] {
  const names = new Set<string>();
  for (const epoch of history) {
    for (const key of Object.keys(epoch)) if (key !== "epoch" && !key.startsWith("val_")) names.add(key);
  }
  return [...names];
}

export function lineTrace(name: string, color: string, history: Epoch[], key: string): Data {
  return {
    type: "scatter",
    mode: "lines",
    name,
    x: history.map((e) => e.epoch),
    y: history.map((e) => e[key] ?? null),
    line: { color, width: 2 },
    hovertemplate: "%{y:.4g}",
  };
}

export const lineLayout = (metric: string): Partial<Layout> => ({
  hovermode: "x unified",
  xaxis: { title: { text: "epoch" } },
  yaxis: { title: { text: metric } },
});
