// In `next dev` the API runs separately on :8000; the static build is served by FastAPI itself.
const API =
  process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "");

export type Dataset = { id: string; name: string; n_rows: number; columns: string[] };

export type Column = {
  name: string;
  kind: "numeric" | "categorical";
  missing: number;
  unique: number;
  mean?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
};

export type Profile = Omit<Dataset, "columns"> & {
  columns: Column[];
  correlation: { columns: string[]; values: (number | null)[][] };
  scatter_sample: Record<string, (number | null)[]>;
};

export type Task = "regression" | "classification";

export type RunConfig = {
  dataset_id: string;
  label: string;
  features: string[];
  task: Task | null;
  name: string | null;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  hidden_layers: number[];
  optimizer: "rmsprop" | "adam" | "sgd";
  validation_split: number;
};

export type Status = "queued" | "running" | "completed" | "cancelled" | "failed";
export const TERMINAL: Status[] = ["completed", "cancelled", "failed"];

export type Epoch = { epoch: number } & Record<string, number>;

export type Run = {
  id: string;
  name: string;
  created_at: number;
  status: Status;
  config: RunConfig;
  task?: Task;
  classes?: string[] | null;
  metrics?: Record<string, number | null>;
  error?: string;
  history?: Epoch[];
};

export type Predictions = {
  features: Record<string, (number | string | null)[]>;
  observed: (number | string)[];
  predicted: (number | string)[];
};

type ValidationError = { loc: (string | number)[]; msg: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(API + path, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail: string | ValidationError[] | undefined = body?.detail;
    throw new Error(
      Array.isArray(detail)
        ? detail.map((e) => `${e.loc.slice(1).join(".")}: ${e.msg}`).join("; ")
        : (detail ?? response.statusText),
    );
  }
  return response.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  listDatasets: () => request<Dataset[]>("/api/datasets"),
  uploadDataset: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Dataset>("/api/datasets", { method: "POST", body: form });
  },
  profile: (id: string) => request<Profile>(`/api/datasets/${id}/profile`),
  listRuns: () => request<Run[]>("/api/runs"),
  getRun: (id: string) => request<Run>(`/api/runs/${id}`),
  createRun: (config: RunConfig) => request<Run>("/api/runs", json(config)),
  cancelRun: (id: string) => request<{ ok: boolean }>(`/api/runs/${id}/cancel`, { method: "POST" }),
  predictions: (id: string) => request<Predictions>(`/api/runs/${id}/predictions`),
  streamUrl: (id: string) => `${API}/api/runs/${id}/stream`,
};

export const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function formatNumber(value: number | null | undefined): string {
  if (value == null) return "—";
  return Math.abs(value) >= 1e5 || (value !== 0 && Math.abs(value) < 1e-3)
    ? value.toExponential(2)
    : Number(value.toPrecision(4)).toString();
}
