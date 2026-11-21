"""Draft validator (stub: checks cited IDs exist in hits)."""


def check_draft(reply: str, cited_ids: list[str], hits: list[dict], ticket_text: str) -> dict:
    known = {h["ticket_id"] for h in hits}
    issues = [f"unknown citation {c}" for c in cited_ids if c not in known]
    if not cited_ids:
        issues.append("no citations")
    return {"grounded": not issues, "cited_ids": [c for c in cited_ids if c in known], "issues": issues}
