"""
Depth Anything V3 reconstruction method implementation.

This wraps the Depth Anything V3 model for multi-view metric depth reconstruction
with pose estimation and sky segmentation support.

Model options:
    - DA3NESTED-GIANT-LARGE (1.40B): Full features including metric depth & sky segmentation
    - DA3-GIANT (1.15B): Relative depth with pose estimation and Gaussian Splatting
    - DA3-LARGE (0.35B): Relative depth with pose estimation
    - DA3METRIC-LARGE (0.35B): Monocular metric depth with sky segmentation
    - DA3MONO-LARGE (0.35B): Monocular relative depth with sky segmentation
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..base import BaseReconstructionMethod, ReconstructionOutput

# Add reconstruction_repositories to path for depth_anything_3 imports
# The depth_anything_3 repo uses internal imports like "from depth_anything_3.cfg import ..."
# So we need to add the PARENT directory (reconstruction_repositories) to sys.path
_REPO_ROOT = Path(__file__).parents[3]  # Rewis3d_Reconstruction
_REPOS_PATH = _REPO_ROOT / "reconstruction_repositories"
if str(_REPOS_PATH) not in sys.path:
    sys.path.insert(0, str(_REPOS_PATH))


class DepthAnythingV3Method(BaseReconstructionMethod):
    """
    Depth Anything V3-based 3D reconstruction method.

    Uses Depth Anything V3 for metric depth estimation with pose estimation,
    confidence scores, and sky segmentation for outdoor scenes.

    Config options (under reconstruction key):
        - model_name: HuggingFace model name (default: "depth-anything/DA3NESTED-GIANT-LARGE")
        - confidence_percentile: Percentile for confidence thresholding (default: 30)

    Available models:
        - "depth-anything/DA3NESTED-GIANT-LARGE": Best for outdoor with sky (1.40B params)
        - "depth-anything/DA3-GIANT": Multi-view with GS support (1.15B params)
        - "depth-anything/DA3-LARGE": Multi-view lightweight (0.35B params)
        - "depth-anything/DA3-BASE": Multi-view small (0.12B params, Apache 2.0)
        - "depth-anything/DA3-SMALL": Multi-view tiny (0.08B params, Apache 2.0)
        - "depth-anything/DA3METRIC-LARGE": Monocular metric depth (0.35B params)
        - "depth-anything/DA3MONO-LARGE": Monocular relative depth (0.35B params)
    """

    name = "depth_anything_v3"

    # Default model - best for outdoor scenes with metric depth and sky segmentation
    DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"

    def __init__(self, device_id: Optional[int] = None, **kwargs):
        """
        Initialize Depth Anything V3 method.

        Args:
            device_id: GPU device ID. If None, uses default CUDA device or CPU.
            **kwargs: Additional configuration options.
        """
        super().__init__(device_id, **kwargs)

        # Model configuration
        self.model_name = kwargs.get("model_name", self.DEFAULT_MODEL)

    def load_model(self) -> None:
        """Load the Depth Anything V3 model from HuggingFace."""
        if self._model is not None:
            return

        # Import here to avoid loading depth_anything_3 unless this method is used
        from depth_anything_3.api import DepthAnything3

        print(
            f"Loading Depth Anything V3 model '{self.model_name}' on device: {self.device}..."
        )
        self._model = DepthAnything3.from_pretrained(self.model_name)
        self._model = self._model.to(device=torch.device(self.device))
        print(f"Depth Anything V3 model loaded on device: {self.device}")

    def reconstruct(
        self, image_paths: List[str], config: Dict[str, Any]
    ) -> ReconstructionOutput:
        """
        Perform 3D reconstruction using Depth Anything V3.

        Args:
            image_paths: List of paths to input images.
            config: Configuration dictionary. Reads from config["reconstruction"].

        Returns:
            ReconstructionOutput with per-view predictions containing:
                - pts3d: 3D points computed from depth and camera intrinsics
                - conf: Confidence scores
                - mask: Valid depth mask
                - img_no_norm: Original images normalized to [0, 1]
                - depth: Raw depth values
                - extrinsics: Camera extrinsics (w2c)
                - intrinsics: Camera intrinsics
        """
        # Ensure model is loaded
        self.load_model()

        # Get reconstruction config
        recon_config = config.get("reconstruction", {})
        confidence_percentile = recon_config.get("confidence_percentile", 30)

        # Override model if specified in config
        model_name = recon_config.get("model_name", self.model_name)
        if model_name != self.model_name:
            # Need to reload model with different weights
            self._model = None
            self.model_name = model_name
            self.load_model()

        # Run inference - Depth Anything V3 takes a list of image paths directly
        prediction = self._model.inference(image_paths)

        # Convert to standardized format
        predictions = self._convert_to_standard_format(prediction, image_paths)

        metadata = {
            "method": self.name,
            "device": self.device,
            "confidence_percentile": confidence_percentile,
            "num_views": len(image_paths),
            "model_name": self.model_name,
            "has_metric_depth": "METRIC" in self.model_name
            or "NESTED" in self.model_name,
            "has_sky_segmentation": "NESTED" in self.model_name
            or "METRIC" in self.model_name
            or "MONO" in self.model_name,
        }

        return ReconstructionOutput(predictions=predictions, metadata=metadata)

    def _convert_to_standard_format(
        self, prediction: Any, image_paths: List[str]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Convert Depth Anything V3 predictions to standardized format.

        The standard format expects per-view dicts with:
            - pts3d: (H, W, 3) 3D points
            - conf: (H, W) confidence
            - mask: (H, W) valid mask
            - img_no_norm: (H, W, 3) RGB in [0, 1]

        Args:
            prediction: Raw Depth Anything V3 prediction object with:
                - processed_images: [N, H, W, 3] uint8
                - depth: [N, H, W] float32
                - conf: [N, H, W] float32
                - extrinsics: [N, 3, 4] float32 (w2c)
                - intrinsics: [N, 3, 3] float32
            image_paths: Original image paths.

        Returns:
            List of per-view prediction dicts.
        """
        num_views = prediction.depth.shape[0]
        predictions = []

        for i in range(num_views):
            depth = prediction.depth[i]  # (H, W)
            conf = prediction.conf[i]  # (H, W)
            img = prediction.processed_images[i]  # (H, W, 3) uint8
            intrinsics = prediction.intrinsics[i]  # (3, 3)
            extrinsics = prediction.extrinsics[i]  # (3, 4)

            H, W = depth.shape

            # Create valid mask (depth > 0 and finite)
            mask = (depth > 0) & np.isfinite(depth)

            # Unproject depth to 3D points using intrinsics
            pts3d = self._depth_to_points(depth, intrinsics)

            # Normalize image to [0, 1]
            img_normalized = img.astype(np.float32) / 255.0

            pred_dict = {
                "pts3d": pts3d,  # (H, W, 3)
                "conf": conf,  # (H, W)
                "mask": mask.astype(np.float32),  # (H, W)
                "img_no_norm": img_normalized,  # (H, W, 3)
                # Additional fields specific to Depth Anything V3
                "depth": depth,  # (H, W) - raw metric depth
                "depth_z": torch.from_numpy(
                    depth
                ),  # (H, W) - for compatibility with save_dataset
                "extrinsics": extrinsics,  # (3, 4) - camera extrinsics (w2c)
                "intrinsics": intrinsics,  # (3, 3) - camera intrinsics
            }

            predictions.append(pred_dict)

        return predictions

    def _depth_to_points(self, depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        Unproject depth map to 3D points using camera intrinsics.

        Args:
            depth: (H, W) depth map in meters.
            intrinsics: (3, 3) camera intrinsic matrix.

        Returns:
            pts3d: (H, W, 3) 3D points in camera coordinates.
        """
        H, W = depth.shape

        # Extract intrinsic parameters
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Create pixel coordinate grids
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        # Unproject to 3D
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts3d = np.stack([x, y, z], axis=-1)  # (H, W, 3)

        return pts3d
