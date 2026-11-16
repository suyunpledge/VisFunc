"""A shared function representation for SVSG images and MotionSpec video evidence."""

from .bridge import attach_recognition, export_motion_spec
from .runtime import VideoRuntime, evaluate

__all__ = ["VideoRuntime", "attach_recognition", "evaluate", "export_motion_spec"]
