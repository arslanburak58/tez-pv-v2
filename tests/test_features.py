"""STAGE-3 sonrası: fiziksel öznitelik testleri."""
import pandas as pd
import pytest


@pytest.mark.skip(reason="STAGE-3'te implement edilecek")
def test_cos_zenith_range() -> None:
    """cos_zenith ∈ [-1, 1], gece negatif, gündüz pozitif."""
    pass


@pytest.mark.skip(reason="STAGE-3'te implement edilecek")
def test_k_t_range() -> None:
    """k_t ∈ [0, 1.5] (clear-sky ratio)."""
    pass
