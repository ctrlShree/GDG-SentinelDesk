"""Mask personal data before the ticket goes anywhere (triage, search, Gemini).

Plain regex, on purpose: asking an LLM to find PII would mean sending it the PII.
Order matters, e.g. emails go first so their digits aren't read as phone numbers.
Ticket IDs (TKT-00042), version numbers (2.4.1), times and short numbers are left alone.
Free-form postal addresses and names are NOT caught; that would need an NER model.
"""
import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_IPV6 = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b|\b(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4})?\b")
# +49 170 1234567, +1 (617) 555-1234, 0170/1234567, 030 12345678, (617) 555-1234
_PHONE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s./-]?)?(?:\(\d{1,5}\)[\s./-]?)?\d[\d\s./-]{6,}\d(?![\w.])")
_ACCOUNT = re.compile(
    r"\b(?:account|acct|customer|kunden|konto)[\s-]*(?:no|nr|number|nummer|id)?\.?[\s:#-]*\d{5,}\b", re.I)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def _phone_like(m: re.Match) -> bool:
    s = m.group(0)
    digits = re.sub(r"\D", "", s)
    if not 8 <= len(digits) <= 15:
        return False
    if re.fullmatch(r"\d{1,4}([./-])\d{1,2}\1\d{1,4}", s):  # dates like 21.11.2026
        return False
    return True


def mask_pii(text: str) -> tuple[str, list[str]]:
    if not text:
        return text or "", []
    kinds: list[str] = []

    def sub(pattern, tag, s, keep=lambda m: True):
        hit = False

        def repl(m):
            nonlocal hit
            if not keep(m):
                return m.group(0)
            hit = True
            return f"[{tag}]"

        out = pattern.sub(repl, s)
        if hit and tag not in kinds:
            kinds.append(tag)
        return out

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_IBAN, "IBAN", text, lambda m: sum(c.isdigit() for c in m.group(0)) >= 10)
    text = sub(_CARD, "CARD", text, lambda m: _luhn_ok(re.sub(r"\D", "", m.group(0))))
    text = sub(_IPV4, "IP", text)
    text = sub(_IPV6, "IP", text)
    text = sub(_ACCOUNT, "ACCOUNT", text)
    text = sub(_PHONE, "PHONE", text, _phone_like)
    return text, kinds
