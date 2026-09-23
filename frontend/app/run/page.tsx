"use client";

import type { Data } from "plotly.js-dist-min";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import Plot, { seriesColor } from "@/components/Plot";
import { api, errorMessage, formatNumber, TERMINAL, type Epoch, type Predictions, type Run } from "@/lib/api";
import { lineLayout, lineTrace, metricNames } from "@/lib/charts";

const PREDICTION_TABLE_ROWS = 50;

function RunView({ id }: { id: string }) {
  const [run, setRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<Epoch[]>([]);
  const [predictions, setPredictions] = useState<Predictions | null>(null);
  const [metric, setMetric] = useState("loss");
  const [error, setError] = useState<string | null>(null);

  // Stream epochs over SSE (the server replays from epoch 1), then load the final run + predictions.
  useEffect(() => {
    const loadFinal = () =>
      api
        .getRun(id)
        .then((r) => {
          setRun(r);
          setHistory(r.history ?? []);
          if (r.status !== "failed") api.predictions(id).then(setPredictions).catch(() => setPredictions(null));
        })
        .catch((e) => setError(errorMessage(e)));

    api.getRun(id).then(setRun).catch((e) => setError(errorMessage(e)));
    const source = new EventSource(api.streamUrl(id));
    source.onmessage = (message) => {
      const epoch: Epoch = JSON.parse(message.data);
      setHistory((current) => [...current.filter((e) => e.epoch !== epoch.epoch), epoch]);
      setRun((current) => (current && current.status === "queued" ? { ...current, status: "running" } : current));
    };
    source.addEventListener("end", () => {
      source.close();
      loadFinal();
    });
    source.onerror = () => {
      // EventSource retries on its own; only give up if the run itself is gone.
      api.getRun(id).catch((e) => {
        source.close();
        setError(errorMessage(e));
      });
    };
    return () => source.close();
  }, [id]);

  const metrics = metricNames(history);
  const chart = useMemo(
    (): Data[] => [
      lineTrace("train", seriesColor(0), history, metric),
      lineTrace("validation", seriesColor(1), history, `val_${metric}`),
    ],
    [history, metric],
  );

  if (error) return <p className="error">{error}</p>;
  if (!run) return <p className="muted">Loading…</p>;

  const config = run.config;
  const running = !TERMINAL.includes(run.status);

  return (
    <>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>
            {run.name} <span className={`status status-${run.status}`}>{run.status}</span>
          </h2>
          {running && (
            <button onClick={() => api.cancelRun(id).catch((e) => setError(errorMessage(e)))}>Cancel</button>
          )}
        </div>
        <p className="muted">
          {config.label} ← {config.features.join(", ")} · {run.task ?? config.task ?? "auto task"} · {config.optimizer} lr{" "}
          {config.learning_rate} · batch {config.batch_size} · {config.epochs} epochs · hidden [
          {config.hidden_layers.join(", ")}] · val {config.validation_split}
        </p>
        {run.error && <p className="error">{run.error}</p>}
      </section>

      <div className="grid-2">
        <section className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h2 style={{ margin: 0 }}>Training curve</h2>
            {metrics.length > 1 && (
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                {metrics.map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            )}
          </div>
          {history.length ? (
            <Plot data={chart} layout={lineLayout(metric)} />
          ) : (
            <p className="muted">{running ? "Waiting for the first epoch…" : "No epochs recorded."}</p>
          )}
        </section>

        <section className="card">
          <h2>Validation metrics</h2>
          {run.metrics ? (
            <table>
              <tbody>
                {Object.entries(run.metrics).map(([name, value]) => (
                  <tr key={name}>
                    <th>{name}</th>
                    <td>{formatNumber(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">Available when training finishes.</p>
          )}
        </section>
      </div>

      {predictions && <PredictionsView predictions={predictions} regression={run.task === "regression"} />}
    </>
  );
}

function PredictionsView({ predictions, regression }: { predictions: Predictions; regression: boolean }) {
  const { observed, predicted, features } = predictions;
  const chart = useMemo((): Data[] => {
    const values = [...observed, ...predicted] as number[];
    const [low, high] = [Math.min(...values), Math.max(...values)];
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "validation rows",
        x: observed,
        y: predicted,
        marker: { color: seriesColor(0), size: 8, opacity: 0.6, line: { color: "var(--surface)", width: 1 } },
        hovertemplate: "observed %{x:.4g}<br>predicted %{y:.4g}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "lines",
        name: "perfect prediction",
        x: [low, high],
        y: [low, high],
        line: { color: "var(--muted)", width: 1 },
        hoverinfo: "skip",
      },
    ];
  }, [observed, predicted]);

  const names = Object.keys(features);
  return (
    <section className="card">
      <h2>
        Predictions <span className="muted">· validation split</span>
      </h2>
      {regression && (
        <Plot
          data={chart}
          height={380}
          layout={{ xaxis: { title: { text: "observed" } }, yaxis: { title: { text: "predicted" } } }}
        />
      )}
      <div className="table-wrap" style={{ maxHeight: 360, overflowY: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>observed</th>
              <th>predicted</th>
              {regression && <th>abs error</th>}
              {names.map((n) => (
                <th key={n}>{n}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {observed.slice(0, PREDICTION_TABLE_ROWS).map((value, i) => (
              <tr key={i}>
                <td>{regression ? formatNumber(value as number) : value}</td>
                <td>{regression ? formatNumber(predicted[i] as number) : predicted[i]}</td>
                {regression && <td>{formatNumber(Math.abs((value as number) - (predicted[i] as number)))}</td>}
                {names.map((n) => (
                  <td key={n}>{typeof features[n][i] === "number" ? formatNumber(features[n][i] as number) : features[n][i]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RunFromQuery() {
  const id = useSearchParams().get("id");
  return id ? <RunView key={id} id={id} /> : <p className="error">No run id given.</p>;
}

export default function RunPage() {
  return (
    <main>
      <Suspense fallback={<p className="muted">Loading…</p>}>
        <RunFromQuery />
      </Suspense>
    </main>
  );
}
