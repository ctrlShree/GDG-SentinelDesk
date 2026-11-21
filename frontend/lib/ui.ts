export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export const priorityTone: Record<string, string> = {
  critical: "bg-red-600 text-white ring-red-700",
  high: "bg-red-100 text-red-800 ring-red-300",
  medium: "bg-amber-100 text-amber-800 ring-amber-300",
  low: "bg-emerald-100 text-emerald-800 ring-emerald-300",
  very_low: "bg-slate-100 text-slate-700 ring-slate-300",
};

export const pct = (x: number) => `${Math.round(x * 100)}%`;

export const fmtMs = (ms: number) => (ms < 10 ? ms.toFixed(1) : Math.round(ms).toLocaleString());
