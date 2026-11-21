"use client";
import { useState } from "react";
import type { PipelineResult } from "@/lib/types";
import Card from "./Card";

const CITE_SPLIT = /(\[TKT-\d{5}\])/g;
const IS_CITE = /^\[TKT-\d{5}\]$/;

function Tag({ children, tone }: { children: React.ReactNode; tone: string }) {
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${tone}`}>{children}</span>;
}

export default function DraftPanel({ r }: { r: PipelineResult }) {
  const [copied, setCopied] = useState(false);
  const d = r.draft;

  const copy = async () => {
    if (!d) return;
    try {
      await navigator.clipboard.writeText(d.reply);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked */ }
  };

  const tags = (
    <div className="flex flex-wrap gap-1">
      {r.draft_for_human_review && <Tag tone="bg-amber-50 text-amber-800 ring-amber-300">For human review</Tag>}
      {d?.fallback && <Tag tone="bg-slate-100 text-slate-700 ring-slate-300">Template fallback</Tag>}
      {d && !d.grounded && <Tag tone="bg-red-50 text-red-700 ring-red-300">Not grounded</Tag>}
      {r.pii_masked && <Tag tone="bg-violet-50 text-violet-800 ring-violet-300">PII masked</Tag>}
      {r.truncated && <Tag tone="bg-slate-100 text-slate-700 ring-slate-300">Truncated to 4,000 chars</Tag>}
    </div>
  );

  return (
    <Card title="Draft reply" right={tags}>
      {!d ? (
        <p className="rounded-md bg-slate-50 p-4 text-slate-600">No draft: escalated before generation.</p>
      ) : (
        <>
          <p className="whitespace-pre-wrap leading-relaxed text-slate-900" lang={r.language}>
            {d.reply.split(CITE_SPLIT).map((part, i) =>
              IS_CITE.test(part) ? (
                <span key={i} className="mx-0.5 rounded-full bg-sky-100 px-2 py-0.5 font-mono text-sm font-semibold text-sky-800">
                  {part.slice(1, -1)}
                </span>
              ) : (
                <span key={i}>{part}</span>
              ),
            )}
          </p>
          {d.issues.length > 0 && (
            <ul className="mt-3 list-disc pl-6 text-sm text-red-700">
              {d.issues.map((x, i) => <li key={i}>{x}</li>)}
            </ul>
          )}
          <div className="mt-3 flex items-center justify-between gap-2 text-sm text-slate-500">
            <span>Cites: {d.cited_ids.length ? d.cited_ids.join(", ") : "none"}</span>
            <button onClick={copy} className="rounded border border-slate-300 px-3 py-1 hover:bg-slate-100">
              {copied ? "Copied ✓" : "Copy"}
            </button>
          </div>
        </>
      )}
      <details className="mt-4 text-sm">
        <summary className="cursor-pointer text-slate-500">Text sent to models{r.pii_masked ? " (PII masked)" : ""}</summary>
        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-3 text-xs">{r.masked_text}</pre>
      </details>
    </Card>
  );
}
