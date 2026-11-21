"""Plain-Python checks that a draft only uses facts from the retrieved tickets. No LLM."""
import re

ID_RE = re.compile(r"TKT-\d{5}")

# money, percentages, durations: "50%", "$20", "20 EUR", "3 days", "48 Stunden"
NUM_RE = re.compile(
    r"(?:[$€]\s?\d+(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)?\s?(?:%|€|\$|EUR|USD|hours?|days?|weeks?|months?|"
    r"Stunden|Tagen?|Wochen|Monaten?))",
    re.IGNORECASE,
)

# word stems that signal a promise (EN + DE)
COMMIT_STEMS = ["guarantee", "refund", "compensat", "credit", "legally",
                "garanti", "erstatt", "entschädig", "gutschrift"]


def _norm(s):
    return re.sub(r"\s+", "", s.lower()).replace(",", ".")


def check_draft(reply, cited_ids, hits, ticket_text):
    """Return {"grounded", "cited_ids", "issues"}."""
    reply = reply or ""
    cited_ids = list(cited_ids or [])
    issues = []
    hit_ids = {h["ticket_id"] for h in hits}

    # 1. Citations: every ID (in the text or the list) must be one of the hits
    in_text = ID_RE.findall(reply)
    all_ids = list(dict.fromkeys(cited_ids + in_text))
    kept = [i for i in all_ids if i in hit_ids]
    unknown = [i for i in all_ids if i not in hit_ids]
    if unknown:
        issues.append("unknown citation: " + ", ".join(unknown))
    if not kept:
        issues.append("no valid citation")

    # 2. Numbers: must appear in the hit texts or the ticket itself
    evidence = " ".join(f"{h.get('subject', '')} {h.get('body', '')} {h.get('answer', '')}" for h in hits)
    evidence_norm = _norm(evidence + " " + (ticket_text or ""))
    for num in NUM_RE.findall(reply):
        if _norm(num) not in evidence_norm:
            issues.append(f"unsupported number: {num.strip()}")

    # 3. Commitments: promise words must appear in at least one hit answer
    answers = " ".join(h.get("answer", "") or "" for h in hits).lower()
    low = reply.lower()
    bad = [w for w in COMMIT_STEMS if w in low and w not in answers]
    if bad:
        issues.append("unsupported commitment: " + ", ".join(bad))

    return {"grounded": not issues, "cited_ids": kept, "issues": issues}
