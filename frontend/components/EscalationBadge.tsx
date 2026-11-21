import type { Escalation } from "@/lib/types";
import { cx } from "@/lib/ui";

export default function EscalationBadge({ e }: { e: Escalation }) {
  const esc = e.status === "ESCALATE";
  return (
    <div className={cx("rounded-xl border-2 p-4", esc ? "border-red-500 bg-red-50" : "border-emerald-500 bg-emerald-50")}>
      <p className={cx("text-2xl font-extrabold", esc ? "text-red-700" : "text-emerald-700")}>
        {esc ? "🚨 Escalate to On-Call Engineer" : "✅ Auto-Draft Ready"}
      </p>
      {e.reasons.length > 0 ? (
        <ul className="mt-2 list-disc space-y-1 pl-6 text-slate-800">
          {e.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      ) : (
        !esc && <p className="mt-1 text-sm text-emerald-800">All gate checks passed: confident, precedented, grounded.</p>
      )}
    </div>
  );
}
