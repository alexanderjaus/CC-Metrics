import os
import shutil
from typing import Dict, List, Tuple

import pytest
import torch
from torch import Tensor

from CCMetrics import CCDiceMetric, CCHausdorffDistance95Metric, CCSurfaceDiceMetric


@pytest.fixture(
    params=[
        # Simple case
        {"size": (64, 64, 64), "offset": (1, 1, 1)},
        # Smaller case
        {"size": (32, 32, 32), "offset": (0, 0, 0)},
    ]
)
def sample_data(request) -> Tuple[Tensor, Tensor]:
    """
    Create sample data for testing with different sizes and offsets.

    Args:
        request: Pytest request object containing parameters

    Returns:
        Tuple of (ground truth, prediction) tensors
    """
    size = request.param["size"]
    offset = request.param["offset"]

    y = torch.zeros((1, 2, *size))
    y_hat = torch.zeros((1, 2, *size))

    # Create a simple cube in ground truth
    y[0, 1, 20:25, 20:25, 20:25] = 1
    y[0, 0] = 1 - y[0, 1]

    # Create a slightly offset cube in prediction
    ox, oy, oz = offset
    y_hat[0, 1, 20 + ox : 25 + ox, 20 + oy : 25 + oy, 20 + oz : 25 + oz] = 1
    y_hat[0, 0] = 1 - y_hat[0, 1]

    return y, y_hat


@pytest.fixture(scope="module")
def cache_dir() -> str:
    """
    Create and manage a cache directory for testing.
    Uses a test-specific directory to avoid conflicts with existing caches.

    Returns:
        Path to cache directory
    """
    cache = ".test_metrics_cache"  # Use a specific test cache directory
    try:
        os.makedirs(cache, exist_ok=True)
        yield cache
    finally:
        if os.path.exists(cache):
            try:
                shutil.rmtree(cache)
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not remove test cache directory: {e}")


def test_cc_dice_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Dice metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCDiceMetric(use_caching=True, caching_dir=cache_dir)
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Dice score should be a float"
    assert 0 <= result <= 1, "Dice score should be between 0 and 1"


def test_cc_surface_dice_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Surface Dice metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCSurfaceDiceMetric(
        use_caching=True, caching_dir=cache_dir, class_thresholds=[1]
    )
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Surface Dice score should be a float"
    assert 0 <= result <= 1, "Surface Dice score should be between 0 and 1"


def test_cc_hausdorff_metric_validity(
    sample_data: Tuple[Tensor, Tensor], cache_dir: str
) -> None:
    """
    Test the Connected Components Hausdorff Distance metric.

    Args:
        sample_data: Tuple of ground truth and predicted tensors
        cache_dir: Directory for caching results
    """
    y, y_hat = sample_data
    metric = CCHausdorffDistance95Metric(
        use_caching=True, caching_dir=cache_dir, metric_worst_score=30
    )
    metric(y_pred=y_hat, y=y)
    result = metric.cc_aggregate().mean().item()

    assert isinstance(result, float), "Hausdorff distance should be a float"
    assert result >= 0, "Hausdorff distance should be non-negative"
    assert result <= 30, "Hausdorff distance should not exceed worst score"


def test_empty_input(cache_dir: str) -> None:
    """
    Test metrics with empty input tensors.

    Args:
        cache_dir: Directory for caching results
    """
    empty_y = torch.zeros((1, 2, 16, 16, 16))
    empty_y_hat = torch.zeros((1, 2, 16, 16, 16))

    metrics: Dict = {
        "dice": CCDiceMetric(use_caching=True, caching_dir=cache_dir),
        "surface_dice": CCSurfaceDiceMetric(
            use_caching=True, caching_dir=cache_dir, class_thresholds=[1]
        ),
        "hd95": CCHausdorffDistance95Metric(
            use_caching=True, caching_dir=cache_dir, metric_worst_score=30
        ),
    }

    for name, metric in metrics.items():
        metric(y_pred=empty_y_hat, y=empty_y)
        result = metric.cc_aggregate().mean().item()
        assert isinstance(result, float), f"{name} should handle empty input"


@pytest.fixture
def multi_patient_data() -> Tuple[List[Tensor], List[Tensor]]:
    """
    Create test data for multiple patients, each with different numbers of components.
    Returns ground truth and prediction tensors for two patients:
    - Patient 1: Two components
    - Patient 2: One component

    Note: Each patient is processed individually due to batch size=1 limitation.

    Returns:
        Tuple of (ground truth list, prediction list) where each list contains
        tensors for individual patients
    """
    shape = (1, 2, 32, 32, 32)

    # Patient 1 data
    y1 = torch.zeros(shape)
    y_hat1 = torch.zeros(shape)

    # Two components for patient 1
    y1[0, 1, 5:10, 5:10, 5:10] = 1  # First component
    y1[0, 1, 20:25, 20:25, 20:25] = 1  # Second component
    y1[0, 0] = 1 - y1[0, 1]

    y_hat1[0, 1, 6:11, 6:11, 6:11] = 1  # Offset first component
    y_hat1[0, 1, 21:26, 21:26, 21:26] = 1  # Offset second component
    y_hat1[0, 0] = 1 - y_hat1[0, 1]

    # Patient 2 data
    y2 = torch.zeros(shape)
    y_hat2 = torch.zeros(shape)

    # One component for patient 2
    y2[0, 1, 15:20, 15:20, 15:20] = 1
    y2[0, 0] = 1 - y2[0, 1]

    y_hat2[0, 1, 16:21, 16:21, 16:21] = 1
    y_hat2[0, 0] = 1 - y_hat2[0, 1]

    return [y1, y2], [y_hat1, y_hat2]


def test_cc_dice_aggregation_modes(
    cache_dir: str, multi_patient_data: Tuple[List[Tensor], List[Tensor]]
) -> None:
    """
    Test CCDiceMetric aggregation modes and numerical correctness.

    Tests two key aspects:
    1. Aggregation behavior: 'patient' vs 'overall' reduction modes
    2. Numerical correctness: Dice scores for 5x5x5 cubes with 1-voxel offset
       Expected score = 2|X∩Y|/(|X|+|Y|) = 2(64)/(125+125) = 0.512
    """
    y_list, y_hat_list = multi_patient_data
    expected_dice = 0.512  # theoretical value for 1-voxel offset
    tolerance = 0.01

    # Test both aggregation modes with same data
    metric_patient = CCDiceMetric(
        use_caching=True, caching_dir=cache_dir, cc_reduction="patient"
    )
    metric_overall = CCDiceMetric(
        use_caching=True, caching_dir=cache_dir, cc_reduction="overall"
    )

    for y, y_hat in zip(y_list, y_hat_list):
        metric_patient(y_pred=y_hat, y=y)
        metric_overall(y_pred=y_hat, y=y)

    # Get per-patient scores
    patient_scores = metric_patient.cc_aggregate().tolist()

    # Get per-component scores
    component_scores = metric_overall.cc_aggregate().tolist()

    print(f"Patient-level scores: {patient_scores}")
    print(f"Component-level scores: {component_scores}")

    # Test 1: Verify aggregation behavior
    assert len(patient_scores) == 2, "Should have one score per patient"
    assert len(component_scores) == 3, "Should have one score per component"

    # Test 2: Verify numerical correctness
    assert all(
        abs(score - expected_dice) < tolerance
        for score in patient_scores + component_scores
    ), "All scores should match theoretical value"
