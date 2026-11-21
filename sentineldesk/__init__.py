"""SentinelDesk: autonomous incident triage & grounded resolution copilot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TARGETS = ("type", "queue", "priority")
