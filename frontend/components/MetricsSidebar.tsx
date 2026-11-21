import Card from "./Card";

type M = Record<string, unknown>;

// metrics.json comes from A's scripts/evaluate.py; accept a few plausible layouts
// until the final keys are agreed.
function num(m: M, ...paths: string[]): number | null {
  for (const p of paths) {
    let cur: unknown = m;
    for (const k of p.split(".")) cur = cur && typeof cur === "object" ? (cur as M)[k] : undefined;
    if (typeof cur === "number") return cur;
  }
  return null;
}

const fmt = (x: number | null) => (x === null ? "—" : x.toFixed(3));
const fmtPct = (x: number | null) => (x === null ? "—" : `${(x <= 1 ? x * 100 : x).toFixed(1)}%`);

export default function MetricsSidebar({ m }: { m: M | null }) {
  if (!m || m.available === false) {
    return (
      <Card title="Holdout metrics">
        <p className="text-slate-500">Not computed yet. Run <code>python scripts/evaluate.py</code>.</p>
      </Card>
    );
  }
  const p95 = num(m, "latency_ms.p95", "latency.p95", "pipeline.latency_ms.total.p95", "pipeline.p95_latency_ms", "p95_latency_ms");
  const n = num(m, "n_test", "test_size", "n");
  const unseen = (t: string) => num(m, `macro_f1_unseen_text.${t}`);
  const rows: [string, string][] = [
    ...(["type", "queue", "priority"] as const).map((t): [string, string] => [
      `Macro-F1 · ${t}`,
      fmt(num(m, `macro_f1.${t}`, `classifier.${t}.macro_f1`, `${t}.macro_f1`, `classification.${t}.macro_f1`)) +
        (unseen(t) === null ? "" : ` (unseen ${fmt(unseen(t))})`),
    ]),
    ["High-priority incident recall", fmtPct(num(m, "high_priority_incident_recall", "hp_incident_recall",
      "pipeline.high_priority_incident_recall", "gate.high_priority_incident_recall"))],
    ["Classifier high-priority recall", fmtPct(num(m, "high_priority_recall.recall", "high_priority_recall"))],
    ["Escalation rate", fmtPct(num(m, "escalation_rate", "pipeline.escalation_rate", "gate.escalation_rate"))],
    ["p95 latency", p95 === null ? "—" : `${Math.round(p95)} ms`],
  ];
  return (
    <Card title="Holdout metrics">
      <dl className="space-y-2">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2 border-b border-slate-100 pb-1 text-sm">
            <dt className="text-slate-600">{k}</dt>
            <dd className="font-semibold tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-xs text-slate-500">
        20% holdout split{n ? ` · ${n.toLocaleString()} tickets` : ""}. &ldquo;Unseen&rdquo; excludes test tickets whose
        text also appears in the KB. Test answers and labels are never indexed or used for training.
      </p>
    </Card>
  );
}
