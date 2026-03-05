"""
Base class for reconstruction methods.

All reconstruction methods must inherit from BaseReconstructionMethod and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ReconstructionOutput:
    """
    Standardized output format for all reconstruction methods.

    Each view in the predictions list should contain:
        - pts3d: 3D points array of shape (H, W, 3)
        - conf: Confidence scores of shape (H, W)
        - mask: Valid point mask of shape (H, W)
        - img_no_norm: Original image in [0, 1] range of shape (H, W, 3)

    Additional method-specific fields can be included and will be passed through.
    """

    predictions: List[Dict[str, Any]]  # List of per-view prediction dicts
    metadata: Dict[str, Any]  # Method-specific metadata


class BaseReconstructionMethod(ABC):
    """
    Abstract base class for 3D reconstruction methods.

    All reconstruction methods should inherit from this class and implement
    the required methods. This ensures a consistent interface across different
    reconstruction approaches (MapAnything, COLMAP, NeRF-based, etc.).

    Example usage:
        class MyReconstructionMethod(BaseReconstructionMethod):
            def __init__(self, device_id=None, **kwargs):
                super().__init__(device_id)
                # Initialize your model here

            def reconstruct(self, image_paths, config):
                # Perform reconstruction
                return ReconstructionOutput(predictions=[...], metadata={...})
    """

    # Method name - should be overridden by subclasses
    name: str = "base"

    def __init__(self, device_id: Optional[int] = None, **kwargs):
        """
        Initialize the reconstruction method.

        Args:
            device_id: GPU device ID. If None, uses default CUDA device or CPU.
            **kwargs: Additional method-specific arguments.
        """
        self.device_id = device_id
        self._model = None
        self._config = kwargs

    @property
    def device(self) -> str:
        """Get the device string for PyTorch."""
        import torch

        if self.device_id is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return f"cuda:{self.device_id}"

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the reconstruction model.

        This method should initialize self._model with the loaded model.
        Called lazily on first reconstruction call.
        """
        pass

    @abstractmethod
    def reconstruct(
        self, image_paths: List[str], config: Dict[str, Any]
    ) -> ReconstructionOutput:
        """
        Perform 3D reconstruction from input images.

        Args:
            image_paths: List of paths to input images for reconstruction.
            config: Configuration dictionary containing reconstruction parameters.

        Returns:
            ReconstructionOutput containing per-view predictions and metadata.

        The predictions list should have one entry per input image, each containing:
            - pts3d: numpy array of shape (H, W, 3) with 3D point coordinates
            - conf: numpy array of shape (H, W) with confidence scores
            - mask: numpy array of shape (H, W) with valid point mask (bool or float)
            - img_no_norm: numpy array of shape (H, W, 3) with RGB values in [0, 1]
        """
        pass

    def preprocess_images(self, image_paths: List[str]) -> Any:
        """
        Preprocess images before reconstruction.

        Override this method if your method needs custom preprocessing.

        Args:
            image_paths: List of paths to input images.

        Returns:
            Preprocessed data in format expected by reconstruct().
        """
        return image_paths

    def postprocess_predictions(
        self, raw_predictions: Any
    ) -> List[Dict[str, np.ndarray]]:
        """
        Convert raw model predictions to standardized format.

        Override this method if your model outputs in a different format.

        Args:
            raw_predictions: Raw output from the reconstruction model.

        Returns:
            List of prediction dicts in standardized format.
        """
        return raw_predictions

    def cleanup(self) -> None:
        """
        Clean up resources (e.g., free GPU memory).

        Override if your method needs custom cleanup logic.
        """
        if self._model is not None:
            del self._model
            self._model = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
