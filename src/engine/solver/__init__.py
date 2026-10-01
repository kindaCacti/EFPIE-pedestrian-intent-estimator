"""Task solvers; detector solver is loaded only when detection is requested."""

from .. import DETECTOR_ROOT

from .tasks import TASKS
from .trajectory_solver import TrajectorySolver

__path__.append(str(DETECTOR_ROOT / "engine" / "solver"))
