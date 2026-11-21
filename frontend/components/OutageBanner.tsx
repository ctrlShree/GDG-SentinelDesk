"use client";
import { useState } from "react";
import type { Burst } from "@/lib/types";

export default function OutageBanner({ burst }: { burst: Burst }) {
  const [open, setOpen] = useState(false);
  if (!burst?.outage) return null;
  return (
    <div role="alert" className="border-b-4 border-red-900 bg-red-600 px-6 py-3 text-white">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3">
        <p className="text-lg font-bold">
          <span className="mr-1 inline-block animate-pulse">🚨</span>
          MAJOR INCIDENT OUTAGE · {burst.cluster_size} similar tickets · {burst.label}
        </p>
        <button onClick={() => setOpen(!open)} className="rounded bg-red-800 px-3 py-1 text-sm hover:bg-red-900">
          {open ? "Hide" : "Show"} tickets
        </button>
      </div>
      {open && (
        <ul className="mx-auto mt-2 flex max-w-7xl flex-wrap gap-2 font-mono text-sm">
          {burst.members.map((m) => (
            <li key={m} className="rounded bg-red-800 px-2 py-0.5">{m}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
