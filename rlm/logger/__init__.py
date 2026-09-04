from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlm.logger.rlm_logger import RLMLogger

if TYPE_CHECKING:
    from rlm.logger.verbose import VerbosePrinter

__all__ = ["RLMLogger", "VerbosePrinter"]


def __getattr__(name: str) -> Any:
    if name == "VerbosePrinter":
        from rlm.logger.verbose import VerbosePrinter

        return VerbosePrinter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
