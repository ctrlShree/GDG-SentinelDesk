"""One helper for all Gemini calls. Returns None on ANY problem so callers can fall back."""
import concurrent.futures
import json
import os
import time

from dotenv import load_dotenv

from sentineldesk import ROOT

load_dotenv(ROOT / ".env")  # repo-root .env, whatever the working directory
DEFAULT_MODEL = "gemini-3.6-flash"  # used only if GEMINI_MODEL is unset/empty in .env
_client = None


def _get_client():
    global _client
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:  # no key -> fallback path
        return None
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def generate_json(system, prompt, schema, timeout_s=20):
    """Call Gemini in JSON mode. Returns a dict, or None on missing key / error / timeout / bad JSON."""
    try:
        client = _get_client()
        if client is None:
            return None
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
        )
        # main model first, then backups (comma-separated in .env)
        models = [(os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()] + [
            m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
        ]
        models = list(dict.fromkeys(models))  # drop duplicates, keep order

        deadline = time.monotonic() + timeout_s  # timeout_s is the TOTAL budget across all models
        for model in models:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = pool.submit(client.models.generate_content, model=model, contents=prompt, config=config)
                try:
                    resp = future.result(timeout=remaining)
                finally:
                    pool.shutdown(wait=False)

                text = (resp.text or "").strip()
                if text.startswith("```"):
                    text = text.strip("`").removeprefix("json").strip()
                out = json.loads(text)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue  # busy / error -> try next model
        return None  # all models failed -> template fallback
    except Exception:
        return None
