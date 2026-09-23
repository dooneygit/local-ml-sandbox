"use client";

import type { Data, Layout } from "plotly.js-dist-min";
import { useEffect, useRef, useState } from "react";

/** Categorical color for series i, in the fixed slot order (never cycled; callers cap at 8). */
export const seriesColor = (i: number) => `var(--series-${i + 1})`;

/** Plotly can't read CSS variables, so swap `var(--x)` strings for their computed values. */
function resolveVars<T>(value: T, style: CSSStyleDeclaration): T {
  if (typeof value === "string" && value.startsWith("var(")) {
    return style.getPropertyValue(value.slice(4, -1)).trim() as T;
  }
  if (Array.isArray(value)) {
    return (typeof value[0] === "number" ? value : value.map((v) => resolveVars(v, style))) as T;
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, resolveVars(v, style)])) as T;
  }
  return value;
}

const axis = { gridcolor: "var(--grid)", linecolor: "var(--axis)", zerolinecolor: "var(--axis)", automargin: true };

export default function Plot({ data, layout, height = 340 }: { data: Data[]; layout?: Partial<Layout>; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [scheme, setScheme] = useState(0); // bumps on light/dark switch so colors re-resolve

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setScheme((n) => n + 1);
    media.addEventListener("change", onChange);
    return () => media.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    let stale = false;
    import("plotly.js-dist-min").then(({ default: Plotly }) => {
      if (stale || !ref.current) return;
      const style = getComputedStyle(document.documentElement);
      const fullLayout: Partial<Layout> = {
        height,
        margin: { t: 32, r: 16, b: 48, l: 56 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "var(--ink-2)", family: "system-ui, -apple-system, 'Segoe UI', sans-serif", size: 12 },
        legend: { orientation: "h", y: 1.12 },
        hoverlabel: { bgcolor: "var(--surface)", bordercolor: "var(--axis)", font: { color: "var(--ink)" } },
        ...layout,
        xaxis: { ...axis, ...layout?.xaxis },
        yaxis: { ...axis, ...layout?.yaxis },
      };
      Plotly.react(ref.current, resolveVars(data, style), resolveVars(fullLayout, style), {
        responsive: true,
        displaylogo: false,
      });
    });
    return () => {
      stale = true;
    };
  }, [data, layout, height, scheme]);

  useEffect(() => {
    const element = ref.current;
    return () => {
      if (element) import("plotly.js-dist-min").then(({ default: Plotly }) => Plotly.purge(element));
    };
  }, []);

  return <div ref={ref} style={{ minHeight: height }} />;
}
