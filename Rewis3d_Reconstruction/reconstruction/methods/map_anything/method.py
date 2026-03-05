# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
MapAnything reconstruction method implementation.

This wraps the MapAnything model from Meta for multi-view 3D reconstruction.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..base import BaseReconstructionMethod, ReconstructionOutput

# Add reconstruction_repositories to path for mapanything imports
_REPO_ROOT = Path(__file__).parents[3]  # Rewis3d_Reconstruction
_MAPANYTHING_PATH = _REPO_ROOT / "reconstruction_repositories" / "mapanything"
if str(_MAPANYTHING_PATH) not in sys.path:
    sys.path.insert(0, str(_MAPANYTHING_PATH))


class MapAnythingMethod(BaseReconstructionMethod):
    """
    MapAnything-based 3D reconstruction method.

    Uses Meta's MapAnything model for dense multi-view 3D reconstruction
    with confidence estimation and masking.

    Config options (under reconstruction key):
        - confidence_percentile: Percentile for confidence thresholding (default: 30)
        - memory_efficient: Use memory efficient inference mode (default: True)
        - use_amp: Use automatic mixed precision (default: True)
        - amp_dtype: AMP dtype, "bf16" or "fp16" (default: "bf16")
        - apply_mask: Apply edge masking (default: True)
        - mask_edges: Mask image edges (default: True)
        - apply_confidence_mask: Apply confidence-based masking (default: False)
        - pretrained_path: HuggingFace model path (default: "facebook/map-anything")
    """

    name = "map_anything"

    def __init__(self, device_id: Optional[int] = None, **kwargs):
        """
        Initialize MapAnything method.

        Args:
            device_id: GPU device ID. If None, uses default CUDA device or CPU.
            **kwargs: Additional configuration options.
        """
        super().__init__(device_id, **kwargs)

        # MapAnything-specific defaults
        self.pretrained_path = kwargs.get("pretrained_path", "facebook/map-anything")
        self.memory_efficient = kwargs.get("memory_efficient", True)
        self.use_amp = kwargs.get("use_amp", True)
        self.amp_dtype = kwargs.get("amp_dtype", "bf16")
        self.apply_mask = kwargs.get("apply_mask", True)
        self.mask_edges = kwargs.get("mask_edges", True)
        self.apply_confidence_mask = kwargs.get("apply_confidence_mask", False)

    def load_model(self) -> None:
        """Load the MapAnything model from HuggingFace."""
        if self._model is not None:
            return

        # Import here to avoid loading mapanything unless this method is used
        from mapanything.models import MapAnything

        print(f"Loading MapAnything model on device: {self.device}...")
        self._model = MapAnything.from_pretrained(self.pretrained_path).to(self.device)
        print(f"MapAnything model loaded on device: {self.device}")

    def preprocess_images(self, image_paths: List[str]) -> Any:
        """
        Load and preprocess images using MapAnything's image loader.

        Args:
            image_paths: List of paths to input images.

        Returns:
            Preprocessed views ready for model inference.
        """
        from mapanything.utils.image import load_images

        return load_images(image_paths)

    def reconstruct(
        self, image_paths: List[str], config: Dict[str, Any]
    ) -> ReconstructionOutput:
        """
        Perform 3D reconstruction using MapAnything.

        Args:
            image_paths: List of paths to input images.
            config: Configuration dictionary. Reads from config["reconstruction"].

        Returns:
            ReconstructionOutput with per-view predictions.
        """
        # Ensure model is loaded
        self.load_model()

        # Get reconstruction config with defaults
        recon_config = config.get("reconstruction", {})
        confidence_percentile = recon_config.get("confidence_percentile", 30)

        # Override instance defaults with config if provided
        memory_efficient = recon_config.get("memory_efficient", self.memory_efficient)
        use_amp = recon_config.get("use_amp", self.use_amp)
        amp_dtype = recon_config.get("amp_dtype", self.amp_dtype)
        apply_mask = recon_config.get("apply_mask", self.apply_mask)
        mask_edges = recon_config.get("mask_edges", self.mask_edges)
        apply_confidence_mask = recon_config.get(
            "apply_confidence_mask", self.apply_confidence_mask
        )

        # Preprocess images
        views = self.preprocess_images(image_paths)

        # Run inference
        raw_predictions = self._model.infer(
            views,
            memory_efficient_inference=memory_efficient,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            apply_confidence_mask=apply_confidence_mask,
        )

        # Convert to standardized format
        predictions = self.postprocess_predictions(raw_predictions)

        metadata = {
            "method": self.name,
            "device": self.device,
            "confidence_percentile": confidence_percentile,
            "num_views": len(image_paths),
            "pretrained_path": self.pretrained_path,
        }

        return ReconstructionOutput(predictions=predictions, metadata=metadata)

    def postprocess_predictions(
        self, raw_predictions: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Convert MapAnything predictions to standardized numpy format.

        Args:
            raw_predictions: List of prediction dicts with torch tensors.

        Returns:
            List of prediction dicts with numpy arrays.
        """
        processed = []
        for pred in raw_predictions:
            processed_pred = {}
            for key, value in pred.items():
                if isinstance(value, torch.Tensor):
                    processed_pred[key] = value.squeeze().detach().cpu().numpy()
                else:
                    processed_pred[key] = value
            processed.append(processed_pred)
        return processed
