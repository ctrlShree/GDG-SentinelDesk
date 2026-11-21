"use client";
import { useCallback, useEffect, useState } from "react";
import DraftPanel from "@/components/DraftPanel";
import EscalationBadge from "@/components/EscalationBadge";
import LatencyBreakdown from "@/components/LatencyBreakdown";
import MetricsSidebar from "@/components/MetricsSidebar";
import OutageBanner from "@/components/OutageBanner";
import PrecedentsTable from "@/components/PrecedentsTable";
import PredictionTiles from "@/components/PredictionTiles";
import TicketForm from "@/components/TicketForm";
import * as api from "@/lib/api";
import type { Burst, Example, Health, PipelineResult } from "@/lib/types";
import { cx } from "@/lib/ui";

export default function Console() {
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [burst, setBurst] = useState<Burst>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  const refreshStatic = useCallback(async () => {
    try {
      const [h, ex, m] = await Promise.all([api.getHealth(), api.getExamples(), api.getMetrics()]);
      setHealth(h); setExamples(ex); setMetrics(m); setBackendDown(false);
    } catch {
      setHealth(null); setBackendDown(true);
    }
  }, []);

  useEffect(() => {
    refreshStatic();
    const id = setInterval(async () => {
      try { setHealth(await api.getHealth()); setBackendDown(false); } catch { setHealth(null); setBackendDown(true); }
    }, 15000);
    return () => clearInterval(id);
  }, [refreshStatic]);

  const run = async (fn: () => Promise<void>) => {
    setLoading(true); setError(""); setNote("");
    try { await fn(); setBackendDown(false); }
    catch (e) {
      const err = e as api.ApiError;
      setError(err.message);
      if (err.status === 0) setBackendDown(true);
    } finally { setLoading(false); }
  };

  const onSubmit = (subject: string, body: string) => run(async () => {
    const r = await api.triage(subject, body);
    setResult(r);
    setBurst(r.burst);
  });

  const onSimulate = () => run(async () => {
    const s = await api.simulateBurst();
    setResult(s.results[s.results.length - 1]);
    setBurst(s.burst);
    setNote(`Simulated ${s.results.length} VPN-failure tickets (EN + DE).` +
      (s.burst?.outage ? "" : " No outage cluster detected."));
  });

  const onReset = () => run(async () => {
    await api.resetBurst();
    setBurst(null); setResult(null);
    setNote("Burst window cleared.");
  });

  const healthy = health?.status === "ok" && health.models_loaded;

  return (
    <div className="min-h-screen bg-slate-100">
      <OutageBanner burst={burst} />
      <header className="bg-slate-900 text-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">SentinelDesk</h1>
            <p className="text-sm text-slate-300">Incident triage &amp; grounded resolution copilot</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            {health && !health.llm_available && (
              <span className="rounded bg-amber-500/20 px-2 py-0.5 text-amber-200">LLM: fallback mode</span>
            )}
            <span className="flex items-center gap-2">
              <span className={cx("h-3 w-3 rounded-full", healthy ? "bg-emerald-400" : "bg-red-500")} />
              {healthy ? "API online" : health ? "Models not loaded" : "API offline"}
            </span>
          </div>
        </div>
      </header>

      {backendDown && (
        <div className="bg-red-100 px-6 py-2 text-center text-red-800">
          Backend unreachable. Start it with <code className="font-mono">uvicorn api:app --port 8000</code>.{" "}
          <button onClick={refreshStatic} className="font-semibold underline">Retry</button>
        </div>
      )}

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 py-6 sm:px-6 lg:grid-cols-[380px_1fr]">
        <aside className="space-y-4">
          <TicketForm examples={examples} loading={loading} onSubmit={onSubmit} onSimulate={onSimulate} onReset={onReset} />
          <MetricsSidebar m={metrics} />
        </aside>

        <div className="min-w-0 space-y-4">
          {error && <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-red-800">{error}</div>}
          {note && <div className="rounded-lg border border-slate-300 bg-white p-3 text-slate-700">{note}</div>}
          {!result ? (
            <div className="flex h-64 items-center justify-center rounded-xl border-2 border-dashed border-slate-300 text-slate-500">
              {loading ? "Triaging…" : "Pick a preset or paste a ticket, then press Triage."}
            </div>
          ) : (
            <div className={cx("space-y-4 transition-opacity", loading && "opacity-50")}>
              <div className="flex items-center justify-between text-sm text-slate-500">
                <span className="font-mono">{result.ticket_id}</span>
              </div>
              <PredictionTiles r={result} />
              <EscalationBadge e={result.escalation} />
              <DraftPanel r={result} />
              <PrecedentsTable hits={result.hits} cited={result.draft?.cited_ids ?? []} />
              <LatencyBreakdown ms={result.latency_ms} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
