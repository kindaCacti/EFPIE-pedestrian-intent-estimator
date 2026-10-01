"""Trajectory extension; existing SOVA detector modules are reused read-only.

The sibling detector checkout remains untouched. Appending its module path
lets adapters/core YAML code keep their original relative imports.
"""

from pathlib import Path
import sys

# Avoid creating bytecode files in the sibling checkout when importing it.
sys.dont_write_bytecode = True

DETECTOR_ROOT = Path(__file__).resolve().parents[1] / "SOVA-signs-det-dev"
if (DETECTOR_ROOT / "engine").is_dir():
    __path__.append(str(DETECTOR_ROOT / "engine"))
