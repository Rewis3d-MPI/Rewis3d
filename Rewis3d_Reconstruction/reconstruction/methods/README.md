# Reconstruction Methods

This folder contains pluggable reconstruction method implementations. Each method is self-contained in its own subfolder and implements the `BaseReconstructionMethod` interface.

## Available Methods

| Method | Description | Folder |
|--------|-------------|--------|
| `map_anything` | Meta's MapAnything dense multi-view reconstruction | `map_anything/` |
| `depth_anything_v3` | Depth Anything V3 with metric depth, pose estimation, and sky segmentation | `depth_anything_v3/` |

## External Repositories

The actual model implementations are stored in `reconstruction_repositories/`:

```
reconstruction_repositories/
├── mapanything/        # MapAnything model code
└── depth_anything_3/   # Depth Anything V3 model code
```

These are git-ignored and should be cloned separately.

## Adding a New Reconstruction Method

### 1. Create the Method Folder

```bash
mkdir reconstruction/methods/your_method
```

### 2. Implement the Method Class

Create `reconstruction/methods/your_method/method.py`:

```python
from typing import Any, Dict, List, Optional
import numpy as np

from ..base import BaseReconstructionMethod, ReconstructionOutput


class YourMethod(BaseReconstructionMethod):
    """Your reconstruction method implementation."""
    
    name = "your_method"  # This is used in config files
    
    def __init__(self, device_id: Optional[int] = None, **kwargs):
        super().__init__(device_id, **kwargs)
        # Initialize your method-specific settings
    
    def load_model(self) -> None:
        """Load your model."""
        if self._model is not None:
            return
        # Load your model here
        self._model = ...
    
    def reconstruct(
        self, 
        image_paths: List[str], 
        config: Dict[str, Any]
    ) -> ReconstructionOutput:
        """
        Perform reconstruction.
        
        Must return ReconstructionOutput with predictions list containing:
        - pts3d: (H, W, 3) numpy array of 3D points
        - conf: (H, W) numpy array of confidence scores
        - mask: (H, W) numpy array of valid point mask
        - img_no_norm: (H, W, 3) numpy array of RGB in [0, 1]
        """
        self.load_model()
        
        # Your reconstruction logic here
        predictions = [...]
        
        return ReconstructionOutput(
            predictions=predictions,
            metadata={"method": self.name}
        )
```

### 3. Create the `__init__.py`

Create `reconstruction/methods/your_method/__init__.py`:

```python
from .method import YourMethod

__all__ = ["YourMethod"]
```

### 4. Register the Method

Add to `reconstruction/methods/registry.py` in `_ensure_builtin_methods_registered()`:

```python
if "your_method" not in _METHOD_REGISTRY:
    from .your_method import YourMethod
    register_method("your_method")(YourMethod)
```

### 5. Use in Config

```yaml
reconstruction:
  method: "your_method"
  # Your method-specific options
  your_option: value
```

## Output Format

All reconstruction methods must output predictions in this standardized format:

```python
ReconstructionOutput(
    predictions=[
        {
            "pts3d": np.ndarray,      # (H, W, 3) - 3D point coordinates
            "conf": np.ndarray,        # (H, W) - Confidence scores [0, 1]
            "mask": np.ndarray,        # (H, W) - Valid points mask (bool/float)
            "img_no_norm": np.ndarray, # (H, W, 3) - RGB image in [0, 1]
            # Additional method-specific fields are allowed
        },
        # ... one dict per input image
    ],
    metadata={
        "method": "method_name",
        # Additional metadata
    }
)
```

## Method Interface

See `base.py` for the full `BaseReconstructionMethod` interface:

- `load_model()`: Load the reconstruction model (called lazily)
- `reconstruct(image_paths, config)`: Main reconstruction method
- `preprocess_images(image_paths)`: Optional custom preprocessing
- `postprocess_predictions(raw)`: Optional output conversion
- `cleanup()`: Free resources
