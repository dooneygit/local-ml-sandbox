"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { api, errorMessage, type Profile, type Run, type RunConfig } from "@/lib/api";

export default function RunForm({ profile }: { profile: Profile }) {
  const columns = profile.columns;
  const [label, setLabel] = useState(columns[columns.length - 1].name);
  const [features, setFeatures] = useState<string[]>([]);
  const [task, setTask] = useState<"auto" | "regression" | "classification">("auto");
  const [name, setName] = useState("");
  const [learningRate, setLearningRate] = useState("0.001");
  const [batchSize, setBatchSize] = useState("50");
  const [epochs, setEpochs] = useState("20");
  const [hiddenLayers, setHiddenLayers] = useState("");
  const [optimizer, setOptimizer] = useState<RunConfig["optimizer"]>("rmsprop");
  const [validationSplit, setValidationSplit] = useState("0.2");
  const [submitted, setSubmitted] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);

  const chooseLabel = (value: string) => {
    setLabel(value);
    setFeatures((current) => current.filter((f) => f !== value));
  };

  const toggleFeature = (feature: string) =>
    setFeatures((current) => (current.includes(feature) ? current.filter((f) => f !== feature) : [...current, feature]));

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    try {
      const run = await api.createRun({
        dataset_id: profile.id,
        label,
        features,
        task: task === "auto" ? null : task,
        name: name.trim() || null,
        learning_rate: Number(learningRate),
        batch_size: Number(batchSize),
        epochs: Number(epochs),
        hidden_layers: hiddenLayers.split(",").map((s) => s.trim()).filter(Boolean).map(Number),
        optimizer,
        validation_split: Number(validationSplit),
      });
      setSubmitted((current) => [run, ...current]);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  return (
    <section className="card">
      <h2>Train a model</h2>
      <form onSubmit={submit} style={{ display: "grid", gap: 14 }}>
        <div className="row">
          <label className="field">
            Label
            <select value={label} onChange={(e) => chooseLabel(e.target.value)}>
              {columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.kind})
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Task
            <select value={task} onChange={(e) => setTask(e.target.value as typeof task)}>
              <option value="auto">auto-detect</option>
              <option value="regression">regression</option>
              <option value="classification">classification</option>
            </select>
          </label>
          <label className="field">
            Run name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="optional" />
          </label>
        </div>

        <div className="field">
          <span className="muted">Features ({features.length} selected)</span>
          <div className="checks">
            {columns
              .filter((c) => c.name !== label)
              .map((c) => (
                <label key={c.name}>
                  <input type="checkbox" checked={features.includes(c.name)} onChange={() => toggleFeature(c.name)} />
                  {c.name} <span className="tag">{c.kind === "numeric" ? "num" : "cat"}</span>
                </label>
              ))}
          </div>
        </div>

        <div className="row">
          <label className="field">
            Learning rate
            <input type="number" step="any" min="0" value={learningRate} onChange={(e) => setLearningRate(e.target.value)} />
          </label>
          <label className="field">
            Batch size
            <input type="number" min="1" value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
          </label>
          <label className="field">
            Epochs
            <input type="number" min="1" max="1000" value={epochs} onChange={(e) => setEpochs(e.target.value)} />
          </label>
          <label className="field">
            Hidden layers
            <input value={hiddenLayers} onChange={(e) => setHiddenLayers(e.target.value)} placeholder="e.g. 64, 32 (empty = linear)" />
          </label>
          <label className="field">
            Optimizer
            <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as RunConfig["optimizer"])}>
              <option value="rmsprop">RMSprop</option>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
            </select>
          </label>
          <label className="field">
            Validation split
            <input type="number" step="0.05" min="0.05" max="0.95" value={validationSplit} onChange={(e) => setValidationSplit(e.target.value)} />
          </label>
        </div>

        <div className="row">
          <button className="primary" type="submit" disabled={features.length === 0}>
            Queue training run
          </button>
          {error && <span className="error">{error}</span>}
        </div>
      </form>

      {submitted.length > 0 && (
        <p>
          Queued:{" "}
          {submitted.map((run, i) => (
            <span key={run.id}>
              {i > 0 && ", "}
              <Link href={`/run/?id=${run.id}`}>{run.name}</Link>
            </span>
          ))}{" "}
          · <Link href="/runs/">compare runs</Link>
        </p>
      )}
    </section>
  );
}
