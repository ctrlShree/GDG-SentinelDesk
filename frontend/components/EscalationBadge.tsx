import type { Escalation } from "@/lib/types";
import { cx } from "@/lib/ui";

// Only the gate's "Critical:" rule pages on-call; every other escalation goes to human review.
const LEVELS = {
  critical: { label: "🚨 Page On-Call Engineer", box: "border-red-500 bg-red-50", text: "text-red-700" },
  review: { label: "⚠️ Needs Human Review", box: "border-amber-500 bg-amber-50", text: "text-amber-700" },
  auto: { label: "✅ Auto-Draft Ready", box: "border-emerald-500 bg-emerald-50", text: "text-emerald-700" },
} as const;

export default function EscalationBadge({ e }: { e: Escalation }) {
  const esc = e.status === "ESCALATE";
  const level = !esc ? LEVELS.auto : e.reasons.some((r) => r.startsWith("Critical")) ? LEVELS.critical : LEVELS.review;
  return (
    <div className={cx("rounded-xl border-2 p-4", level.box)}>
      <p className={cx("text-2xl font-extrabold", level.text)}>{level.label}</p>
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
