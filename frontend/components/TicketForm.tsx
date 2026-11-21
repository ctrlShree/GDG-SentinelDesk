"use client";
import { useState } from "react";
import type { Example } from "@/lib/types";
import Card from "./Card";

export default function TicketForm({ examples, loading, onSubmit, onSimulate, onReset }: {
  examples: Example[]; loading: boolean;
  onSubmit: (subject: string, body: string) => void; onSimulate: () => void; onReset: () => void;
}) {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [preset, setPreset] = useState("");
  const [error, setError] = useState("");

  const submit = () => {
    if (!body.trim()) { setError("Ticket body is required."); return; }
    setError("");
    onSubmit(subject, body);
  };

  const pick = (id: string) => {
    setPreset(id);
    const ex = examples.find((e) => e.id === id);
    if (ex) { setSubject(ex.subject); setBody(ex.body); setError(""); }
  };

  return (
    <Card title="Incoming ticket">
      <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor="preset">Demo preset</label>
      <select id="preset" value={preset} onChange={(e) => pick(e.target.value)}
        className="mb-3 w-full rounded-md border border-slate-300 bg-white px-3 py-2">
        <option value="" disabled>Choose an example…</option>
        {examples.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
      </select>

      <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor="subject">Subject (optional)</label>
      <input id="subject" value={subject} onChange={(e) => setSubject(e.target.value)}
        className="mb-3 w-full rounded-md border border-slate-300 px-3 py-2" placeholder="e.g. VPN down for all staff" />

      <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor="body">Body</label>
      <textarea id="body" value={body} rows={8} onChange={(e) => setBody(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }}
        className="w-full rounded-md border border-slate-300 px-3 py-2 font-mono text-sm"
        placeholder="Paste the customer's message (English or German)…" />
      <div className="mt-1 flex justify-between gap-2 text-xs text-slate-500">
        <span>{error ? <span className="font-medium text-red-600">{error}</span> : "Ctrl+Enter to triage"}</span>
        <span className={body.length > 4000 ? "font-medium text-amber-700" : ""}>
          {body.length.toLocaleString()} chars{body.length > 4000 && " · truncated to 4,000"}
        </span>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <button onClick={submit} disabled={loading}
          className="flex items-center gap-2 rounded-md bg-slate-900 px-5 py-2 font-semibold text-white hover:bg-slate-700 disabled:opacity-60">
          {loading && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />}
          Triage
        </button>
        <button onClick={onSimulate} disabled={loading}
          className="rounded-md border border-red-300 bg-red-50 px-4 py-2 font-medium text-red-700 hover:bg-red-100 disabled:opacity-60">
          🔁 Simulate burst
        </button>
        <button onClick={onReset} disabled={loading}
          className="rounded-md border border-slate-300 px-4 py-2 text-slate-700 hover:bg-slate-100 disabled:opacity-60">
          Reset
        </button>
      </div>
    </Card>
  );
}
