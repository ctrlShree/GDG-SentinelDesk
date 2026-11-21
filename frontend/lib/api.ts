import type { BurstSimulation, Example, Health, PipelineResult } from "./types";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError("Backend unreachable. Is `uvicorn api:app --port 8000` running?", 0);
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* body wasn't JSON, e.g. the proxy itself failed */
    }
    if (res.status >= 500 && res.status !== 503) detail = `Backend error: ${detail}`;
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

export const triage = (subject: string, body: string) =>
  call<PipelineResult>("triage", { method: "POST", body: JSON.stringify({ subject, body }) });
export const simulateBurst = () => call<BurstSimulation>("burst/simulate", { method: "POST" });
export const resetBurst = () => call<{ ok: boolean }>("burst/reset", { method: "POST" });
export const getExamples = () => call<Example[]>("examples");
export const getMetrics = () => call<Record<string, unknown>>("metrics");
export const getHealth = () => call<Health>("health");
