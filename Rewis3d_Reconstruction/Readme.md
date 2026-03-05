# Rewis3d Reconstruction

A modular 3D reconstruction pipeline for generating point cloud datasets from multi-view images. This module supports pluggable reconstruction methods and is designed for creating training data for 3D semantic segmentation models.

## Overview

The reconstruction pipeline:
1. **Loads images** from a dataset (KITTI-360, Waymo, Cityscapes, NYUv2, etc.)
2. **Generates 3D reconstructions** using configurable methods (MapAnything, Depth Anything V3)
3. **Unprojects 2D labels** to create labeled 3D point clouds
4. **Saves datasets** in `.npz` format for training

## Installation

### Prerequisites
- Python ≥ 3.11
- CUDA-compatible GPU

### Setup with UV (recommended)

```bash
cd Rewis3d_Reconstruction

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Clone reconstruction repositories (not included in repo)
mkdir -p reconstruction_repositories
cd reconstruction_repositories
git clone <mapanything-repo-url> mapanything
git clone <depth-anything-v3-repo-url> depth_anything_3
```

## Quick Start

```bash
# Run reconstruction with a config file
uv run -m reconstruction.generate_dataset --config reconstruction/config/kitti360.yaml

# Multi-GPU support
uv run -m reconstruction.generate_dataset --config reconstruction/config/kitti360.yaml --num_gpus 4 --gpu_ids 0,1,2,3
```

## Configuration

Configuration files are located in `reconstruction/config/`. Example config:

```yaml
dataset:
  dataset_dir: "/path/to/KITTI360/"
  output_dir: "/path/to/output/"
  splits: ["training", "validation"]
  num_cameras: 1
  file_extension: "png"

sampling:
  strategy: "chunk"
  num_images: 100

reconstruction:
  method: "map_anything"  # or "depth_anything_v3"
  confidence_percentile: 10

point_sampling:
  method: "random_radius"
  num_points: 120000
  image_ratio: 1.0

labels:
  ignore_index: 255
  class_definitions:
    - ["road", 7, 0, [128, 64, 128]]
    - ["sidewalk", 8, 1, [244, 35, 232]]
    # ... more classes
```

### Available Configs

| Config | Dataset | Description |
|--------|---------|-------------|
| `kitti360.yaml` | KITTI-360 | Outdoor driving scenes |
| `waymo.yaml` | Waymo Open | Outdoor driving scenes |
| `cityscapes_*.yaml` | Cityscapes | Urban street scenes |
| `nyuv2.yaml` | NYUv2 | Indoor scenes |

## Reconstruction Methods

The pipeline supports multiple reconstruction methods via a pluggable architecture.

### MapAnything (default)

Meta's dense multi-view reconstruction model.

```yaml
reconstruction:
  method: "map_anything"
  confidence_percentile: 10
  memory_efficient: true
  use_amp: true
  amp_dtype: "bf16"
```

### Depth Anything V3

Metric depth estimation with pose estimation and sky segmentation.

```yaml
reconstruction:
  method: "depth_anything_v3"
  model_name: "depth-anything/DA3NESTED-GIANT-LARGE"  # Best for outdoor with sky
  confidence_percentile: 10
```

Available models:
| Model | Params | Metric Depth | Sky Seg | License |
|-------|--------|--------------|---------|---------|
| `DA3NESTED-GIANT-LARGE` | 1.40B | ✅ | ✅ | CC BY-NC 4.0 |
| `DA3-GIANT` | 1.15B | ❌ | ❌ | CC BY-NC 4.0 |
| `DA3METRIC-LARGE` | 0.35B | ✅ | ✅ | Apache 2.0 |

## Project Structure

```
Rewis3d_Reconstruction/
├── reconstruction/
│   ├── __init__.py
│   ├── generate_dataset.py      # Main entry point
│   ├── create_reconstructions.py # Reconstruction dispatcher
│   ├── image_loading.py         # Image loading utilities
│   ├── pointcloud_processing.py # Point cloud creation
│   ├── point_sampling.py        # Sampling strategies
│   ├── save_dataset.py          # Dataset saving
│   ├── label_utils.py           # Label mapping utilities
│   ├── config/                  # Dataset configs
│   │   ├── kitti360.yaml
│   │   ├── waymo.yaml
│   │   └── ...
│   └── methods/                 # Pluggable reconstruction methods
│       ├── base.py              # Abstract base class
│       ├── registry.py          # Method factory
│       ├── map_anything/
│       │   └── method.py
│       └── depth_anything_v3/
│           └── method.py
├── reconstruction_repositories/ # External model repos (gitignored)
│   ├── mapanything/
│   └── depth_anything_3/
├── pyproject.toml
└── uv.lock
```

## Output Format

The pipeline outputs `.npz` files with the following structure:

```python
data = np.load("output.npz", allow_pickle=True)

# 3D point cloud
data["student_coord"]      # (N, 3) float32 - XYZ coordinates
data["student_colors"]     # (N, 3) float32 - RGB colors [0, 1]
data["conf"]               # (N,) float32 - Confidence scores

# Labels (unprojected from 2D)
data["student_segment"]    # (N,) uint8 - Scribble/weak labels
data["original_segment"]   # (N,) uint8 - Full mask labels

# Metadata
data["view_point_mapping"] # Per-view point indices and pixel coords
```

## Adding a New Reconstruction Method

1. Create a new folder: `reconstruction/methods/your_method/`

2. Implement the method class:

```python
# reconstruction/methods/your_method/method.py
from ..base import BaseReconstructionMethod, ReconstructionOutput

class YourMethod(BaseReconstructionMethod):
    name = "your_method"
    
    def load_model(self):
        # Load your model
        pass
    
    def reconstruct(self, image_paths, config):
        # Return ReconstructionOutput with standardized format
        return ReconstructionOutput(
            predictions=[{
                "pts3d": ...,      # (H, W, 3)
                "conf": ...,       # (H, W)
                "mask": ...,       # (H, W)
                "img_no_norm": ... # (H, W, 3)
            }],
            metadata={"method": self.name}
        )
```

3. Register in `reconstruction/methods/registry.py`:

```python
if "your_method" not in _METHOD_REGISTRY:
    from .your_method import YourMethod
    register_method("your_method")(YourMethod)
```

4. Use in config:

```yaml
reconstruction:
  method: "your_method"
```

## API Usage

```python
from reconstruction import generate_reconstruction, get_available_methods

# List available methods
print(get_available_methods())  # ['map_anything', 'depth_anything_v3']

# Generate reconstruction
config = {
    "reconstruction": {
        "method": "map_anything",
        "confidence_percentile": 30
    }
}
predictions = generate_reconstruction(config, image_paths, device_id=0)
```

## License

This project is for research purposes. Individual reconstruction methods may have their own licenses:
- MapAnything: See Meta's license
- Depth Anything V3: CC BY-NC 4.0 or Apache 2.0 depending on model
