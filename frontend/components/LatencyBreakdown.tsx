import type { LatencyStage } from "@/lib/types";
import { fmtMs } from "@/lib/ui";
import Card from "./Card";

const STAGES: Exclude<LatencyStage, "total">[] = ["pii", "triage", "retrieval", "generation", "gate"];

export default function LatencyBreakdown({ ms }: { ms: Record<LatencyStage, number> }) {
  const max = Math.max(1, ...STAGES.map((s) => ms[s] ?? 0));
  return (
    <Card title="Latency" right={<span className="font-bold tabular-nums">{fmtMs(ms.total)} ms total</span>}>
      <div className="space-y-1.5">
        {STAGES.map((s) => (
          <div key={s} className="flex items-center gap-2 text-sm">
            <span className="w-20 capitalize text-slate-600">{s === "pii" ? "PII" : s}</span>
            <div className="h-2.5 flex-1 rounded-full bg-slate-100">
              <div className="h-2.5 rounded-full bg-sky-600" style={{ width: `${((ms[s] ?? 0) / max) * 100}%` }} />
            </div>
            <span className="w-16 text-right tabular-nums text-slate-700">{fmtMs(ms[s] ?? 0)} ms</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
