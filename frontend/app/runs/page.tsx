"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, errorMessage, formatNumber, TERMINAL, type Dataset, type Run } from "@/lib/api";
import { MAX_COMPARE } from "@/lib/charts";

function headline(run: Run): string {
  if (!run.metrics) return "—";
  const name = run.task === "classification" ? "accuracy" : "rmse";
  return `${name} ${formatNumber(run.metrics[name])}`;
}

export default function RunsPage() {
  const router = useRouter();
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDatasets()
      .then((list: Dataset[]) => setDatasets(Object.fromEntries(list.map((d) => [d.id, d.name]))))
      .catch((e) => setError(errorMessage(e)));
  }, []);

  // Poll while anything is queued or running so statuses update in place.
  const active = runs.some((r) => !TERMINAL.includes(r.status));
  useEffect(() => {
    const load = () => api.listRuns().then(setRuns).catch((e) => setError(errorMessage(e)));
    load();
    if (!active) return;
    const timer = setInterval(load, 2000);
    return () => clearInterval(timer);
  }, [active]);

  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id) ? current.filter((s) => s !== id) : current.length < MAX_COMPARE ? [...current, id] : current,
    );

  return (
    <main>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
          <h2 style={{ margin: 0 }}>Runs</h2>
          <div className="row">
            <span className="muted">
              {selected.length} selected (max {MAX_COMPARE})
            </span>
            <button
              className="primary"
              disabled={selected.length < 2}
              onClick={() => router.push(`/compare/?ids=${selected.join(",")}`)}
            >
              Compare
            </button>
          </div>
        </div>
        {error && <p className="error">{error}</p>}
        {runs.length === 0 ? (
          <p className="muted">
            No runs yet. <Link href="/">Train one</Link>.
          </p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th />
                  <th>run</th>
                  <th>status</th>
                  <th>dataset</th>
                  <th>label ← features</th>
                  <th>hyperparameters</th>
                  <th>val metric</th>
                  <th>created</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const c = run.config;
                  return (
                    <tr key={run.id}>
                      <td>
                        <input type="checkbox" checked={selected.includes(run.id)} onChange={() => toggle(run.id)} />
                      </td>
                      <td>
                        <Link href={`/run/?id=${run.id}`}>{run.name}</Link>
                      </td>
                      <td className={`status status-${run.status}`}>{run.status}</td>
                      <td>{datasets[c.dataset_id] ?? c.dataset_id}</td>
                      <td>
                        {c.label} ← {c.features.join(", ")}
                      </td>
                      <td className="muted">
                        {c.optimizer} lr {c.learning_rate} · bs {c.batch_size} · {c.epochs} ep · [{c.hidden_layers.join(",")}]
                      </td>
                      <td>{headline(run)}</td>
                      <td className="muted">{new Date(run.created_at * 1000).toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
