from __future__ import annotations

import rlm


def test_root_exports_are_eager_and_introspectable() -> None:
    expected = {
        "RLM",
        "BudgetExceededError",
        "CancellationError",
        "ErrorThresholdExceededError",
        "TimeoutExceededError",
        "TokenLimitExceededError",
    }

    assert expected <= vars(rlm).keys()
    assert expected <= set(dir(rlm))
