import type { Label, PipelineResult } from "@/lib/types";
import { cx, fmtMs, pct, priorityTone } from "@/lib/ui";

function Tile({ name, l, chip }: { name: string; l: Label; chip?: string }) {
  const low = l.confidence < 0.45;
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{name}</p>
      <p className="mt-1 text-xl font-bold text-slate-900">
        {chip ? <span className={cx("rounded-full px-3 py-0.5 ring-1", chip)}>{l.label}</span> : l.label}
      </p>
      <div className="mt-3 h-2 rounded-full bg-slate-100">
        <div className={cx("h-2 rounded-full", low ? "bg-amber-500" : "bg-sky-600")}
          style={{ width: pct(Math.min(1, Math.max(0, l.confidence))) }} />
      </div>
      <p className={cx("mt-1 text-sm", low ? "font-medium text-amber-700" : "text-slate-600")}>
        {pct(l.confidence)} confidence{low && " · low"}
      </p>
    </div>
  );
}

export default function PredictionTiles({ r }: { r: PipelineResult }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Tile name="Type" l={r.triage.type} />
      <Tile name="Queue" l={r.triage.queue} />
      <Tile name="Priority" l={r.triage.priority} chip={priorityTone[r.triage.priority.label] ?? priorityTone.low} />
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Language · Latency</p>
        <p className="mt-1 text-xl font-bold">
          <span className="rounded bg-slate-900 px-2 py-0.5 text-white">{r.language.toUpperCase()}</span>
        </p>
        <p className="mt-3 text-3xl font-bold tabular-nums text-slate-900">
          {fmtMs(r.latency_ms.total)}<span className="text-base font-medium text-slate-500"> ms</span>
        </p>
      </div>
    </div>
  );
}
