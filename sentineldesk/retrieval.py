"""Dense retrieval over the KB (stub: fixed fake hits)."""
import numpy as np

_FAKE = [
    ("TKT-00042", "VPN connection drops", "Our VPN disconnects every few minutes.",
     "Please update the VPN client to the latest version and re-import the profile.", "en", "Incident", "IT Support", "high"),
    ("TKT-01337", "Passwort zurücksetzen", "Ich kann mein Passwort nicht zurücksetzen.",
     "Bitte nutzen Sie den Link 'Passwort vergessen' im Anmeldeportal.", "de", "Request", "Technical Support", "medium"),
    ("TKT-02718", "Cannot log in to portal", "Login fails with an error after the update.",
     "Clear the browser cache and reset your password via the self-service portal.", "en", "Problem", "Technical Support", "medium"),
    ("TKT-03141", "Email sync not working", "Outlook stopped syncing since this morning.",
     "Remove and re-add the account, then restart Outlook.", "en", "Incident", "IT Support", "medium"),
    ("TKT-04669", "Request new laptop setup", "New hire needs laptop provisioned.",
     "We have scheduled the provisioning; the device will be ready within 2 business days.", "en", "Request", "IT Support", "low"),
]
_SCORES = [0.92, 0.89, 0.86, 0.83, 0.80]
_KEYS = ("ticket_id", "subject", "body", "answer", "language", "type", "queue", "priority")


class Retriever:
    dim = 384

    def embed(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        v = rng.normal(size=(len(texts), self.dim)).astype("float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def search(self, text: str, k: int = 5) -> list[dict]:
        hits = [dict(zip(_KEYS, row), score=s) for row, s in zip(_FAKE, _SCORES)]
        return hits[:k]
