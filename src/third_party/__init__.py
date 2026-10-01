"""Optional sources owned by this extension and read-only SOVA sources."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "SOVA-signs-det-dev" / "third_party"))
