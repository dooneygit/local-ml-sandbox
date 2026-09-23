"use client";

import { useEffect, useState, type ChangeEvent } from "react";
import ProfileView from "@/components/ProfileView";
import RunForm from "@/components/RunForm";
import { api, errorMessage, type Dataset, type Profile } from "@/lib/api";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listDatasets()
      .then((list) => {
        setDatasets(list);
        setSelected((current) => current ?? list[0]?.id ?? null);
      })
      .catch((e) => setError(errorMessage(e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    let stale = false;
    api
      .profile(selected)
      .then((p) => !stale && setProfile(p))
      .catch((e) => !stale && setError(errorMessage(e)));
    return () => {
      stale = true;
    };
  }, [selected]);

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setError(null);
    try {
      const dataset = await api.uploadDataset(file);
      setDatasets((current) => [dataset, ...current]);
      setSelected(dataset.id);
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  return (
    <main>
      <section className="card">
        <h2>Dataset</h2>
        <div className="row">
          <label className="field">
            Upload CSV
            <input type="file" accept=".csv,text/csv" onChange={upload} />
          </label>
          {datasets.length > 0 && (
            <label className="field">
              Or pick an uploaded one
              <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value)}>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.n_rows.toLocaleString()} rows)
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {profile && profile.id === selected && (
        <>
          <RunForm key={profile.id} profile={profile} />
          <ProfileView key={`profile-${profile.id}`} profile={profile} />
        </>
      )}
    </main>
  );
}
