"use client";
import { useState } from "react";
import type { Hit } from "@/lib/types";
import Card from "./Card";

export default function PrecedentsTable({ hits, cited }: { hits: Hit[]; cited: string[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <Card title={`Historical precedents (top ${hits.length})`}>
      {hits.length === 0 ? <p className="text-slate-500">No precedents retrieved.</p> : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="border-b text-xs uppercase text-slate-500">
              <tr>
                <th className="py-2 pr-3">Ticket</th><th className="pr-3">Similarity</th>
                <th className="pr-3">Lang</th><th>Historical answer</th>
              </tr>
            </thead>
            <tbody>
              {hits.map((h) => {
                const isOpen = open === h.ticket_id;
                const long = h.answer.length > 200;
                return (
                  <tr key={h.ticket_id} className="border-b align-top last:border-0">
                    <td className="whitespace-nowrap py-2 pr-3 font-mono font-semibold">
                      {h.ticket_id}
                      {cited.includes(h.ticket_id) && (
                        <span className="ml-1 rounded bg-sky-100 px-1 font-sans text-xs text-sky-800">cited</span>
                      )}
                    </td>
                    <td className="py-2 pr-3 tabular-nums">{h.score.toFixed(2)}</td>
                    <td className="py-2 pr-3 uppercase">{h.language}</td>
                    <td className="py-2">
                      {h.subject && <p className="font-medium text-slate-800">{h.subject}</p>}
                      <p className="text-slate-600">{isOpen || !long ? h.answer : h.answer.slice(0, 200) + "…"}</p>
                      {long && (
                        <button onClick={() => setOpen(isOpen ? null : h.ticket_id)}
                          className="text-xs font-medium text-sky-700 hover:underline">
                          {isOpen ? "Show less" : "Show more"}
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
