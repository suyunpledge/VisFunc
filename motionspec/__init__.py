"""motionspec：把视频里的界面动效拆成 AI 读得懂的规格（数值 + 少量标注帧）。"""

from .analyze import analyze
from .claims import check
from .spec import SCHEMA, summarize, validate

__version__ = "0.1.0"
__all__ = ["analyze", "check", "summarize", "validate", "SCHEMA", "__version__"]
