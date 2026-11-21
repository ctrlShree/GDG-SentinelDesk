// TypeScript mirror of the frozen PipelineResult contract (see PLAN.md).
export type Label = { label: string; confidence: number; probs: Record<string, number> };
export type Triage = { type: Label; queue: Label; priority: Label };
export type Hit = {
  ticket_id: string; score: number; subject: string; body: string; answer: string;
  language: string; type: string; queue: string; priority: string;
};
export type Draft = { reply: string; cited_ids: string[]; grounded: boolean; issues: string[]; fallback: boolean };
export type Escalation = { escalate: boolean; status: "AUTO_DRAFT" | "ESCALATE"; reasons: string[] };
export type Burst = { outage: boolean; cluster_size: number; label: string; members: string[] } | null;
export type LatencyStage = "pii" | "triage" | "retrieval" | "generation" | "gate" | "total";
export type PipelineResult = {
  ticket_id: string; language: "en" | "de"; pii_masked: boolean; masked_text: string;
  triage: Triage; hits: Hit[]; draft: Draft | null; draft_for_human_review: boolean;
  escalation: Escalation; burst: Burst; latency_ms: Record<LatencyStage, number>; truncated?: boolean;
};
export type Example = { id: string; label: string; subject: string; body: string };
export type Health = { status: string; models_loaded: boolean; llm_available: boolean };
export type BurstSimulation = { results: PipelineResult[]; burst: Burst };
