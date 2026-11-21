import { cx } from "@/lib/ui";

export default function Card({ title, right, children, className }: {
  title?: string; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={cx("rounded-xl border border-slate-200 bg-white p-4 shadow-sm", className)}>
      {(title || right) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {title && <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}
