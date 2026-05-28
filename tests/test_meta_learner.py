"""STAGE-6 Test Paketi: Meta-Learner FastQuantileRegressor (PyTorch tabanlı)."""

import pathlib
import numpy as np
import pandas as pd
import pytest

from models.meta_learner import (
    FastQuantileRegressor,
    train_meta_learner,
    predict_intervals,
    compute_coverage,
    compute_pinball,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_data():
    """Küçük sentetik OOF verisi."""
    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(
        {
            "xgb_q01": rng.uniform(0.0, 0.4, n),
            "xgb_q50": rng.uniform(0.2, 0.7, n),
            "xgb_q90": rng.uniform(0.5, 1.0, n),
            "lgb_q01": rng.uniform(0.0, 0.4, n),
            "lgb_q50": rng.uniform(0.2, 0.7, n),
            "lgb_q90": rng.uniform(0.5, 1.0, n),
            "cat_q01": rng.uniform(0.0, 0.4, n),
            "cat_q50": rng.uniform(0.2, 0.7, n),
            "cat_q90": rng.uniform(0.5, 1.0, n),
        }
    )
    y = pd.Series(rng.uniform(0.0, 1.0, n), name="y_norm")
    return x, y


# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------

class TestTrainMetaLearner:
    def test_returns_three_models(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y, quantiles=[0.1, 0.5, 0.9])
        assert set(models.keys()) == {0.1, 0.5, 0.9}

    def test_models_are_fitted(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        for q, mdl in models.items():
            assert hasattr(mdl, "coef_"), f"q={q} modeli fit edilmemiş"
            assert mdl.coef_.shape == (x.shape[1],)

    def test_custom_iter_accepted(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y, max_iter=10)
        assert len(models) == 3


class TestPredictIntervals:
    def test_output_shape(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x)
        assert preds.shape == (len(x), 3)
        assert list(preds.columns) == ["q_0.1", "q_0.5", "q_0.9"]

    def test_monotonicity_enforced(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x, enforce_monotonicity=True)
        assert (preds["q_0.1"] <= preds["q_0.5"]).all(), "q_0.1 > q_0.5 crossing var!"
        assert (preds["q_0.5"] <= preds["q_0.9"]).all(), "q_0.5 > q_0.9 crossing var!"

    def test_no_enforce_may_have_crossing(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x, enforce_monotonicity=False)
        assert preds.shape == (len(x), 3)

    def test_predictions_bounded(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x)
        assert preds.abs().max().max() < 10.0, "Tahminler makul sınırın dışında"


class TestMetrics:
    def test_pinball_perfect(self):
        y = np.array([0.5, 0.5, 0.5])
        y_pred = np.array([0.5, 0.5, 0.5])
        assert compute_pinball(y, y_pred, 0.5) == pytest.approx(0.0)

    def test_pinball_positive(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x)
        for q, col in [(0.1, "q_0.1"), (0.5, "q_0.5"), (0.9, "q_0.9")]:
            pb = compute_pinball(y.values, preds[col].values, q)
            assert pb >= 0.0, f"Pinball loss negatif olamaz: q={q}"

    def test_coverage_between_zero_one(self, dummy_data):
        x, y = dummy_data
        models = train_meta_learner(x, y)
        preds = predict_intervals(models, x)
        cov = compute_coverage(y, preds)
        assert 0.0 <= cov <= 1.0
