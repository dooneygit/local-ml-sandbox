"use client";

import type { Data } from "plotly.js-dist-min";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import Plot, { seriesColor } from "@/components/Plot";
import { api, errorMessage, formatNumber, type Run } from "@/lib/api";
import { lineLayout, lineTrace, MAX_COMPARE, metricNames } from "@/lib/charts";

const CONFIG_ROWS: [string, (run: Run) => string][] = [
  ["status", (r) => r.status],
  ["task", (r) => r.task ?? r.config.task ?? "auto"],
  ["label", (r) => r.config.label],
  ["features", (r) => r.config.features.join(", ")],
  ["optimizer", (r) => r.config.optimizer],
  ["learning rate", (r) => String(r.config.learning_rate)],
  ["batch size", (r) => String(r.config.batch_size)],
  ["epochs", (r) => String(r.config.epochs)],
  ["hidden layers", (r) => `[${r.config.hidden_layers.join(", ")}]`],
  ["validation split", (r) => String(r.config.validation_split)],
];

function CompareView({ ids }: { ids: string[] }) {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [metric, setMetric] = useState("loss");
  const [split, setSplit] = useState<"val_" | "">("val_");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(ids.map(api.getRun))
      .then(setRuns)
      .catch((e) => setError(errorMessage(e)));
  }, [ids]);

  const chart = useMemo(
    (): Data[] =>
      (runs ?? []).map((run, i) => lineTrace(run.name, seriesColor(i), run.history ?? [], `${split}${metric}`)),
    [runs, metric, split],
  );

  if (error) return <p className="error">{error}</p>;
  if (!runs) return <p className="muted">Loading…</p>;

  const metrics = [...new Set(runs.flatMap((r) => metricNames(r.history ?? [])))];
  const finalMetrics = [...new Set(runs.flatMap((r) => Object.keys(r.metrics ?? {})))];

  return (
    <>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Training curves</h2>
          <div className="row">
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              {metrics.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
            <select value={split} onChange={(e) => setSplit(e.target.value as typeof split)}>
              <option value="val_">validation</option>
              <option value="">train</option>
            </select>
          </div>
        </div>
        <Plot data={chart} layout={lineLayout(`${split}${metric}`)} height={400} />
      </section>

      <section className="card">
        <h2>Side by side</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th />
                {runs.map((run, i) => (
                  <th key={run.id}>
                    <span className="swatch" style={{ background: seriesColor(i) }} />
                    <Link href={`/run/?id=${run.id}`}>{run.name}</Link>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CONFIG_ROWS.map(([name, value]) => (
                <tr key={name}>
                  <th>{name}</th>
                  {runs.map((run) => (
                    <td key={run.id}>{value(run)}</td>
                  ))}
                </tr>
              ))}
              {finalMetrics.map((name) => (
                <tr key={name}>
                  <th>val {name}</th>
                  {runs.map((run) => (
                    <td key={run.id}>{formatNumber(run.metrics?.[name])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function CompareFromQuery() {
  const param = useSearchParams().get("ids") ?? "";
  const ids = useMemo(() => param.split(",").filter(Boolean).slice(0, MAX_COMPARE), [param]);
  return ids.length ? <CompareView ids={ids} /> : <p className="error">No runs selected.</p>;
}

export default function ComparePage() {
  return (
    <main>
      <Suspense fallback={<p className="muted">Loading…</p>}>
        <CompareFromQuery />
      </Suspense>
    </main>
  );
}
