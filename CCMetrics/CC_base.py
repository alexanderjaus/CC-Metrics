import copy
import gc
import hashlib
import os
from enum import Enum

import numpy as np
import torch
from monai.metrics import (
    Cumulative,
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from torch.nn import functional as F

from CCMetrics.space_separation import compute_voronoi_regions as space_separation


class CCBaseMetric:

    def __init__(
        self,
        BaseMetric: Cumulative,
        *args,
        use_caching=True,
        caching_dir=".cache",
        metric_best_score=None,
        metric_worst_score=None,
        **kwargs,
    ):
        """
        Initializes a CC_base object.

        Args:
            BaseMetric (Cumulative): The base Monai metric to be used.
            *args: Variable length argument list, passed to the Monai metric.
            use_caching (bool, optional): Flag to enable caching. Defaults to True.
            caching_dir (str, optional): Directory to store the cache. Defaults to ".cache".
            metric_best_score: The best score for the metric. Must be defined.
            metric_worst_score: The worst score for the metric. Must be defined.
            **kwargs: Arbitrary keyword arguments, passed to the Monai metric.

        Raises:
            AssertionError: If metric_best_score or metric_worst_score is not defined.

        """
        assert metric_best_score is not None, "Best score must be defined"
        assert metric_worst_score is not None, "Worst score must be defined"
        self.metric_perfect_score = metric_best_score
        self.metric_worst_score = metric_worst_score
        self.buffer_collection = []
        if kwargs.get("include_background", False):
            raise ValueError("Background class is not supported")
        else:
            kwargs["include_background"] = False

        if kwargs.get("cc_reduction", None):
            assert kwargs["cc_reduction"] in [
                "patient",
                "overall",
            ], f"Unknown aggregation function {kwargs['cc_reduction']}"
            self.cc_reduction = kwargs["cc_reduction"]
            del kwargs["cc_reduction"]
        else:
            self.cc_reduction = "patient"

        self.base_metric = BaseMetric(*args, **kwargs)
        self.use_caching = use_caching
        self.caching_dir = caching_dir
        if self.use_caching and not os.path.exists(self.caching_dir):
            os.makedirs(self.caching_dir)

    def __call__(self, y_pred, y):
        """
        Calculates the metric for the predicted and ground truth tensors.

        Args:
            y_pred (numpy.ndarray or torch.Tensor): The predicted tensor.
            y (numpy.ndarray or torch.Tensor): The ground truth tensor.

        Raises:
            AssertionError: If the input shapes or conditions are not correct.

        Returns:
            None
        """
        # Check if tensor or numpy array
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        assert isinstance(
            y_pred, torch.Tensor
        ), f"Input is not a torch tensor. Got {type(y_pred)}"
        assert isinstance(
            y, torch.Tensor
        ), f"Input is not a torch tensor. Got {type(y)}"

        # Check conditions
        assert (
            len(y_pred.shape) == 5
        ), "Input shape is not correct. Expected shape: (B,C,D,H,W) as input y_pred"
        assert (
            len(y.shape) == 5
        ), "Input shape is not correct. Expected shape: (B,C,D,H,W) as input y"
        assert (
            y_pred.shape == y.shape
        ), f"Input shapes do not match. Got {y_pred.shape} and {y.shape}"
        assert (
            y_pred.shape[1] == 2
        ), f"Expected two classes in the input. Got {y_pred.shape[1]}"
        assert y.shape[1] == 2, f"Expected two classes in the input. Got {y.shape[1]}"
        assert (
            y_pred.shape[0] == 1
        ), f"Currently only a batch size of 1 is supported. Got {y_pred.shape[0]} in y_pred"
        assert (
            y.shape[0] == 1
        ), f"Currently only a batch size of 1 is supported. Got {y.shape[0]} in y"
        # Collect from previous runs
        gc.collect()

        # Check if pure backgorund class
        if y[0].argmax(0).sum() == 0:
            if y_pred[0].argmax(0).sum() == 0:
                # Case perfect prediction: No foreground class present in prediction
                self.buffer_collection.append(torch.tensor([self.metric_perfect_score]))
            else:
                # Case worst prediction: Predicted Foreground class but no GT
                self.buffer_collection.append(torch.tensor([self.metric_worst_score]))
            return

        # Get separation as by ground-truth
        if self.use_caching:

            gt_fingerprint = hashlib.md5(
                y[0].argmax(0).cpu().numpy().tobytes()
            ).hexdigest()
            target_path = f"{os.path.join(self.caching_dir, gt_fingerprint)}.npy"
            if os.path.exists(target_path):
                cc_assignment = np.load(target_path)
            else:
                cc_assignment = space_separation(y[0].argmax(0).cpu().numpy())
                np.save(target_path, cc_assignment)
        else:
            cc_assignment = space_separation(y[0].argmax(0).cpu().numpy())

        cc_assignment = torch.from_numpy(cc_assignment).type(torch.int64)

        missed_components = 0

        for cc_id in cc_assignment.unique().tolist():
            pred_helper = copy.deepcopy(y_pred[0]).argmax(0)
            label_helper = copy.deepcopy(y[0]).argmax(0)
            # Find current region of interest
            cc_mask = cc_assignment == cc_id
            pred_helper[torch.logical_not(cc_mask)] = 0
            label_helper[torch.logical_not(cc_mask)] = 0
            if pred_helper.sum() == 0:
                missed_components += 1

            # Remap metrics back to one-hot encoding
            pred_helper = F.one_hot(pred_helper, num_classes=2).permute(3, 0, 1, 2)
            label_helper = F.one_hot(label_helper, num_classes=2).permute(3, 0, 1, 2)

            self.base_metric(
                y_pred=pred_helper.unsqueeze(0), y=label_helper.unsqueeze(0)
            )
            del pred_helper
            del label_helper
            del cc_mask
            gc.collect()

        # Get metric buffer and reset it
        metric_buffer = self.base_metric.get_buffer()
        self.buffer_collection.append(metric_buffer)
        self.base_metric.reset()

    def cc_aggregate(self, mode=None):
        """
        Aggregates the buffer collection based on the specified mode.

        Args:
            mode (str, optional): The aggregation mode. Can be "patient" or "overall".
                If not provided, the default mode specified in self.cc_reduction will be used.

        Returns:
            torch.Tensor: The aggregated result based on the specified mode.

        Raises:
            AssertionError: If an unknown aggregation function is provided.

        """
        if mode is None:
            mode = self.cc_reduction
        assert mode in ["patient", "overall"], f"Unknown aggregation function {mode}"
        cleaned_buffer = [
            torch.where(
                torch.isinf(x),
                torch.tensor(
                    self.metric_worst_score, dtype=torch.float32, device=x.device
                ),
                x,
            )
            for x in self.buffer_collection
        ]
        cleaned_buffer = [
            torch.where(
                torch.isnan(x),
                torch.tensor(
                    self.metric_worst_score, dtype=torch.float32, device=x.device
                ),
                x,
            )
            for x in cleaned_buffer
        ]
        cleaned_buffer = [x.reshape(-1, 1) for x in cleaned_buffer]
        if mode == "patient":
            # Aggregate per patient and return list of means
            return torch.stack([x.mean() for x in cleaned_buffer])
        elif mode == "overall":
            # Aggregate overall. All components are considered as equal. Return full list
            return torch.concatenate(cleaned_buffer).squeeze()

    def get_buffer(self):
        """
        Returns the buffer collection.
        """
        return self.buffer_collection

    def reset(self):
        """
        Resets the buffer collection.
        """
        self.buffer_collection = []

    def cache_datapoint(self, y):
        """
        Caches the datapoint if caching is enabled.

        Args:
            y (torch.Tensor): The input tensor.

        Raises:
            ValueError: If caching is disabled.

        Returns:
            None
        """
        if self.use_caching:
            # Handle data input
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            assert isinstance(
                y, np.ndarray
            ), "Input is not a numpy array or torch tensor. Caching is not possible"
            assert (
                len(y.shape) == 3
            ), "Input shape is not correct. Expected shape: (D,H,W) as input y"

            gt_fingerprint = hashlib.md5(y.tobytes()).hexdigest()
            target_path = f"{os.path.join(self.caching_dir, gt_fingerprint)}.npy"
            if os.path.exists(target_path):
                return
            cc_assignment = space_separation(y)
            np.save(target_path, cc_assignment)
        else:
            raise ValueError("Caching is disabled")


# Define used metrics from the paper <https://arxiv.org/pdf/2410.18684>
# For unbound metrics, the worst score is set to None and should be handled by the user, as it is infinite


class CCDiceMetric(CCBaseMetric):
    """
    CCDiceMetric is a class that represents the Dice metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            DiceMetric, *args, metric_best_score=1.0, metric_worst_score=0.0, **kwargs
        )


class CCHausdorffDistanceMetric(CCBaseMetric):
    """
    CCHausdorffDistanceMetric is a class that represents the Hausdorff distance metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            HausdorffDistanceMetric, *args, metric_best_score=0.0, **kwargs
        )


class CCHausdorffDistance95Metric(CCBaseMetric):
    """
    A class representing a metric for calculating the 95th percentile Hausdorff distance for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            HausdorffDistanceMetric,
            *args,
            metric_best_score=0.0,
            percentile=95,
            **kwargs,
        )


class CCSurfaceDistanceMetric(CCBaseMetric):
    """
    A class representing a metric for calculating the SurfaceDistance metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(SurfaceDistanceMetric, *args, metric_best_score=0.0, **kwargs)


class CCSurfaceDiceMetric(CCBaseMetric):
    """
    A class representing a metric for calculating the SurfaceDiceMetric metric for connected components.
    It inherits from the CCBaseMetric class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            SurfaceDiceMetric,
            *args,
            metric_best_score=1.0,
            metric_worst_score=0.0,
            **kwargs,
        )
