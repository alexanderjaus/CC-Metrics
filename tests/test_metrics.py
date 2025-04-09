import torch
import shutil
import os
import pytest
from CCMetrics import CCDiceMetric, CCSurfaceDiceMetric, CCHausdorffDistance95Metric

@pytest.fixture(scope="module")
def sample_data():
    # Create sample data
    y = torch.zeros((1, 2, 64, 64, 64))
    y_hat = torch.zeros((1, 2, 64, 64, 64))

    y[0, 1, 20:25, 20:25, 20:25] = 1
    y[0, 0] = 1 - y[0, 1]

    y_hat[0, 1, 21:26, 21:26, 21:26] = 1
    y_hat[0, 0] = 1 - y_hat[0, 1]

    return y, y_hat

@pytest.fixture(scope="module")
def cache_dir():
    # Create and clean cache directory
    cache = ".cache"
    if os.path.exists(cache):
        shutil.rmtree(cache)
    os.makedirs(cache, exist_ok=True)
    yield cache
    shutil.rmtree(cache)  # cleanup after tests

def test_metrics(sample_data, cache_dir):
    y, y_hat = sample_data

    # Initialize metrics
    metrics = {
        "dice": CCDiceMetric(use_caching=True, caching_dir=cache_dir),
        "surface_dice": CCSurfaceDiceMetric(use_caching=True, caching_dir=cache_dir, class_thresholds=[1]),
        "hd95": CCHausdorffDistance95Metric(use_caching=True, caching_dir=cache_dir, metric_worst_score=30)
    }

    # Compute and assert metrics
    results = {}
    for name, metric in metrics.items():
        metric(y_pred=y_hat, y=y)
        result = metric.cc_aggregate().mean().item()
        results[name] = result
        assert isinstance(result, float), f"{name} did not return a float"
        assert result >= 0, f"{name} returned a negative score"

    print(f"Results: {results}")
