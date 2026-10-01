"""Task registry for the extension entry point."""

from .trajectory_solver import TrajectorySolver


class _DetectionSolver:
    def __new__(cls, cfg):
        from .det_solver import DetSolver
        return DetSolver(cfg)


TASKS = {"trajectory": TrajectorySolver, "detection": _DetectionSolver}
