"""Grounded reply generation (stub: extractive template citing the hits)."""


def draft_reply(subject: str, body: str, lang: str, hits: list[dict]) -> dict:
    top = hits[:3]
    cited = [h["ticket_id"] for h in top]
    steps = "\n".join(f"- {h['answer']} [{h['ticket_id']}]" for h in top)
    if lang == "de":
        reply = f"Hallo,\n\nvielen Dank für Ihre Nachricht. Basierend auf ähnlichen Fällen:\n{steps}\n\nMit freundlichen Grüßen\nSupport-Team"
    else:
        reply = f"Hello,\n\nThanks for reaching out. Based on similar resolved tickets:\n{steps}\n\nBest regards,\nSupport Team"
    return {"reply": reply, "cited_ids": cited, "grounded": True, "issues": [], "fallback": True}
