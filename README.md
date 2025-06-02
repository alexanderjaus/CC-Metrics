# CC-Metrics
## Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks

[![Paper](https://img.shields.io/badge/PDF-Paper-green.svg)](https://arxiv.org/pdf/2410.18684) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![YouTube](https://img.shields.io/badge/YouTube-Video-red.svg)](https://www.youtube.com/watch?v=VBiXteZkSHs)


## Description
Traditional metrics often fail to adequately capture the performance of models in multi-instance segmentation scenarios, particularly when dealing with heterogeneous structures of varying sizes. CC-Metrics addresses this by:

1. Identifying individual connected components in ground-truth labels
2. Creating Voronoi regions around each component to define its territory
3. Mapping predictions within each Voronoi region to the corresponding ground-truth component
4. Computing standard metrics on these mapped regions for more granular assessment

Below is an example visualization of the Voronoi-based mapping process:

![CC-Metrics Workflow](resources/title_fig.jpg)

For more details, you can read the full paper [here](https://arxiv.org/pdf/2410.18684).


## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [How to Use CC-Metrics](#how-to-use-cc-metrics)
    - [Basic Usage](#basic-usage)
    - [Supported Metrics](#supported-metrics)
    - [Metric Aggregation](#metric-aggregation)
    - [Caching Mechanism](#caching-mechanism)
    - [Advanced Examples](#advanced-examples)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- MONAI 0.9+

```bash
git clone https://github.com/alexanderjaus/CC-Metrics.git
cd CC-Metrics
pip install -e .
```

## How to Use CC-Metrics

CC-Metrics defines wrappers around MONAI's Cumulative metrics to enable per-component evaluation.

### Basic Usage

Here's a simple example using the CCDiceMetric:

```python
from CCMetrics import CCDiceMetric
import torch

# Create the metric with desired parameters
cc_dice = CCDiceMetric(
    cc_reduction="patient",  # Aggregation mode
    use_caching=True,        # Enable caching for faster repeat evaluations
    caching_dir=".cache"     # Directory to store cached Voronoi diagrams
)

# Create sample prediction and ground truth tensors
# Tensors must be in shape (B, C, D, H, W) where:
# B = batch size (currently only B=1 is supported)
# C = number of channels (must be 2: background and foreground)
# D, H, W = depth, height, width of the volumetric data
y = torch.zeros((1, 2, 64, 64, 64))
y_hat = torch.zeros((1, 2, 64, 64, 64))

# Create two ground truth components
y[0, 1, 20:25, 20:25, 20:25] = 1  # Component 1
y[0, 1, 40:45, 40:45, 40:45] = 1  # Component 2
y[0, 0] = 1 - y[0, 1]  # Background

# Create prediction (slightly offset from ground truth)
y_hat[0, 1, 21:26, 21:26, 21:26] = 1  # Prediction for component 1
y_hat[0, 1, 41:46, 39:44, 41:46] = 1  # Prediction for component 2
y_hat[0, 0] = 1 - y_hat[0, 1]  # Background

# Compute the metric
cc_dice(y_pred=y_hat, y=y)

# Get the results
patient_wise_results = cc_dice.cc_aggregate()
#tensor([0.5120])

print(f"CC-Dice score: {patient_wise_results.mean().item()}")

# You can change the scheme during aggregation
component_wise_results = cc_dice.cc_aggregate(mode="overall")
#tensor([0.5120, 0.5120])
```

### Supported Metrics

CC-Metrics includes the following metrics, all derived from MONAI:

- **CCDiceMetric**: Component-wise Dice coefficient
  ```python
  CCDiceMetric()
  ```

- **CCHausdorffDistanceMetric**: Component-wise Hausdorff distance
  ```python
  CCHausdorffDistanceMetric(metric_worst_score=30)
  ```

- **CCHausdorffDistance95Metric**: Component-wise 95th percentile Hausdorff distance
  ```python
  CCHausdorffDistance95Metric(metric_worst_score=30)
  ```

- **CCSurfaceDistanceMetric**: Component-wise average surface distance
  ```python
  CCSurfaceDistanceMetric(metric_worst_score=30)
  ```

- **CCSurfaceDiceMetric**: Component-wise Surface Dice score
  ```python
  CCSurfaceDiceMetric(class_thresholds=[1])
  ```
  This class needs the additional parameter class_thresholds, a list of class-specific thresholds. The thresholds relate to the acceptable amount of deviation in the segmentation boundary in pixels. Each threshold needs to be a finite, non-negative number. More details [here](https://docs.monai.io/en/stable/metrics.html#monai.metrics.SurfaceDiceMetric)


### Metric Aggregation

The `CCBaseMetric` class supports two types of metric aggregation modes:

1. **Patient-Level Aggregation (`patient`)**:
   - Computes the mean metric score for each patient by aggregating all connected components within the patient
   - Returns a list of mean scores, one for each patient
   - Useful when you want to evaluate performance on a per-patient basis

2. **Overall Aggregation (`overall`)**:
   - Treats all connected components across all patients equally
   - Aggregates the metric scores for all components into a single list
   - Useful when you want to evaluate performance across all components regardless of patient boundaries

The aggregation mode can be specified using the `cc_aggregate` method, with the default mode being `patient`.

```python
# Patient-level aggregation (default)
patient_results = cc_dice.cc_aggregate(mode="patient")

# Overall aggregation
overall_results = cc_dice.cc_aggregate(mode="overall")
```

### Caching Mechanism

CC-Metrics requires the computation of a generalized Voronoi diagram which serves as the mapping mechanism between predictions and ground-truth. As the separation of the image space only depends on the ground-truth, the mapping can be cached and reused between intermediate evaluations or across metrics.

#### Benefits of Caching

- Significantly faster repeated evaluations
- Ability to precompute Voronoi regions for large datasets
- Consistent component mapping across different metrics

#### Using the Caching Feature

Enable caching when instantiating any CC-Metrics metric:

```python
cc_dice = CCDiceMetric(use_caching=True, caching_dir="/path/to/cache")
```

#### Precomputing Cache

For large datasets, you can precompute the Voronoi regions using the provided script:

```bash
python prepare_caching.py --gt /path/to/ground_truth_nifti_files --cache_dir /path/to/cache --nof_workers 8
```

This will process all `.nii.gz` files in the specified directory and store the computed Voronoi regions in the cache directory.

### Advanced Examples

#### Evaluating Multiple Metrics on the Same Data

```python
from CCMetrics import CCDiceMetric, CCSurfaceDiceMetric, CCHausdorffDistance95Metric
import torch

# Create sample data
y = torch.zeros((1, 2, 64, 64, 64))
y_hat = torch.zeros((1, 2, 64, 64, 64))

# Set up components (simplified example)
y[0, 1, 20:25, 20:25, 20:25] = 1
y[0, 0] = 1 - y[0, 1]
y_hat[0, 1, 21:26, 21:26, 21:26] = 1
y_hat[0, 0] = 1 - y_hat[0, 1]

# Define shared cache directory
cache_dir = ".cache"

# Initialize metrics
metrics = {
    "dice": CCDiceMetric(use_caching=True, caching_dir=cache_dir),
    "surface_dice": CCSurfaceDiceMetric(use_caching=True, caching_dir=cache_dir, class_thresholds=[1]),
    "hd95": CCHausdorffDistance95Metric(use_caching=True, caching_dir=cache_dir, metric_worst_score=30)
}

# Compute all metrics
results = {}
for name, metric in metrics.items():
    metric(y_pred=y_hat, y=y)
    results[name] = metric.cc_aggregate().mean().item()

print(f"Results: {results}")
```

## FAQ

### Q: Why use CC-Metrics instead of traditional metrics?

A: Traditional metrics like Dice can be misleading in multi-instance segmentation tasks. CC-Metrics provides a more granular assessment of performance by evaluating each component separately, making it particularly valuable for medical imaging tasks with multiple structures of varying sizes.

### Q: How does CC-Metrics handle false negatives (ground truth components with no matching predictions)?

A: CC-Metrics assigns the worst score to false negative regions, ensuring they appropriately penalize the overall performance score.

### Q: How does CC-Metrics handle false positives (predicted components with no matching ground truth)?

A: CC-Metrics evaluates locally thus positive predictions reduce the scores in the region into which they fall.

### Q: Is multi-class segmentation supported?

A: Currently, CC-Metrics only supports binary segmentation (background and foreground). Multi-class support is planned for future releases.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you make use of this project in your work, please cite the CC-Metrics paper:

```bibtex
@article{jaus2024every,
  title={Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks},
  author={Jaus, Alexander and Seibold, Constantin Marc and Rei{\ss}, Simon and Marinov, Zdravko and Li, Keyi and Ye, Zeling and Krieg, Stefan and Kleesiek, Jens and Stiefelhagen, Rainer},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={3904--3912},
  year={2025}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
