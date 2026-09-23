"use client";

import type { Data } from "plotly.js-dist-min";
import { useMemo, useState } from "react";
import { formatNumber, type Profile } from "@/lib/api";
import Plot, { seriesColor } from "./Plot";

const MAX_CELL_LABELS = 12; // beyond this many columns, correlation values live in the tooltip only

export default function ProfileView({ profile }: { profile: Profile }) {
  const numeric = profile.correlation.columns;
  const [dimensions, setDimensions] = useState<string[]>(numeric.slice(0, 4));

  const heatmap = useMemo((): Data[] => {
    const { columns, values } = profile.correlation;
    return [
      {
        type: "heatmap",
        x: columns,
        y: columns,
        z: values,
        zmin: -1,
        zmax: 1,
        // diverging: red (negative) ↔ neutral gray ↔ blue (positive)
        colorscale: [
          [0, "var(--series-8)"],
          [0.5, "var(--diverging-mid)"],
          [1, "var(--series-1)"],
        ],
        texttemplate: columns.length <= MAX_CELL_LABELS ? "%{z:.2f}" : "",
        textfont: { color: "var(--ink)" },
        hovertemplate: "%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        xgap: 2,
        ygap: 2,
      },
    ];
  }, [profile]);

  const splom = useMemo(
    (): Data[] => [
      {
        type: "splom",
        dimensions: dimensions.map((name) => ({ label: name, values: profile.scatter_sample[name] })),
        showupperhalf: false,
        diagonal: { visible: false },
        marker: { color: seriesColor(0), size: 4, opacity: 0.4 },
      } as Data,
    ],
    [profile, dimensions],
  );

  const toggle = (name: string) =>
    setDimensions((current) => (current.includes(name) ? current.filter((d) => d !== name) : [...current, name]));

  return (
    <>
      <section className="card">
        <h2>
          Columns <span className="muted">· {profile.n_rows.toLocaleString()} rows</span>
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {["column", "kind", "missing", "unique", "mean", "std", "min", "max"].map((h) => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {profile.columns.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td className="muted">{c.kind}</td>
                  <td>{c.missing}</td>
                  <td>{c.unique}</td>
                  <td>{formatNumber(c.mean)}</td>
                  <td>{formatNumber(c.std)}</td>
                  <td>{formatNumber(c.min)}</td>
                  <td>{formatNumber(c.max)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {numeric.length > 1 && (
        <div className="grid-2">
          <section className="card">
            <h2>Correlation matrix</h2>
            <Plot
              data={heatmap}
              height={440}
              layout={{ margin: { t: 16, r: 16, b: 16, l: 16 }, yaxis: { autorange: "reversed" } }}
            />
          </section>
          <section className="card">
            <h2>
              Scatter matrix <span className="muted">· sample of {profile.scatter_sample[numeric[0]].length} rows</span>
            </h2>
            <div className="checks">
              {numeric.map((name) => (
                <label key={name}>
                  <input type="checkbox" checked={dimensions.includes(name)} onChange={() => toggle(name)} />
                  {name}
                </label>
              ))}
            </div>
            {dimensions.length > 1 ? (
              <Plot data={splom} height={440} layout={{ margin: { t: 16, r: 16, b: 16, l: 16 }, dragmode: "select" }} />
            ) : (
              <p className="muted">Pick at least two columns.</p>
            )}
          </section>
        </div>
      )}
    </>
  );
}
