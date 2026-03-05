# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
3D Point Cloud Augmentation

Inspirited by chrischoy/SpatioTemporalSegmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
from collections.abc import Sequence, Mapping
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchvision.transforms
from scipy.cluster.hierarchy import correspond

from pointcept.utils.registry import Registry

from torchvision.transforms import (
    ColorJitter,
    RandomCrop,
    RandomHorizontalFlip,
    GaussianBlur,
    AugMix,
)

from transformers import SegformerImageProcessor as SegformerFeatureExtractor

from PIL import Image

from pointcept.utils.logger import get_root_logger

TRANSFORMS2D = Registry("transforms2d")


@TRANSFORMS2D.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, Image.Image):
            data = np.array(data)
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS2D.register_module()
class ToPIL(object):
    def __init__(self, keys=None):
        self.keys = keys or [
            "student_image_1",
            "teacher_image_1",
            "student_mask_1",
            "teacher_mask_1",
            "original_mask_1",
            "instance_mask_original",
            "depth",
            "pseudo_instance",
        ]

    def __call__(self, data_dict):
        for key in self.keys:
            if key in data_dict and isinstance(data_dict[key], np.ndarray):
                data_dict[key] = Image.fromarray(data_dict[key])
        return data_dict


@TRANSFORMS2D.register_module()
class Collect(object):
    def __init__(self, keys, **kwargs):
        """
        e.g. Collect(keys=[image])
        """
        self.keys = keys
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS2D.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS2D.register_module()
class MapIds(object):
    def __init__(self, id2trainId, id2trainId_original_mask=None):
        self.id2trainId = id2trainId
        self.id2trainId_original_mask = id2trainId_original_mask

    def __call__(self, data_dict):
        if self.id2trainId is not None:
            for img_type in ["student_", "teacher_"]:
                mask_array = np.array(data_dict[img_type + "mask_1"])
                mapped_segmentation_map = np.copy(mask_array)
                for original_value, new_value in self.id2trainId.items():
                    mapped_segmentation_map[mask_array == original_value] = new_value
                data_dict[img_type + "mask_1"] = Image.fromarray(
                    mapped_segmentation_map
                )

        if self.id2trainId_original_mask is not None:
            mask_array = np.array(data_dict["original_mask_1"])
            mapped_segmentation_map = np.copy(mask_array)
            for original_value, new_value in self.id2trainId_original_mask.items():
                mapped_segmentation_map[mask_array == original_value] = new_value
            data_dict["original_mask_1"] = Image.fromarray(mapped_segmentation_map)

        return data_dict


@TRANSFORMS2D.register_module()
class ExtractFeatures(object):
    def __init__(self, height=960, width=960, model_name="nvidia/mit-b4"):
        self.feature_extractor = SegformerFeatureExtractor().from_pretrained(model_name)
        self.feature_extractor.size = {"height": height, "width": width}

    def __call__(self, data_dict):
        for img_type in ["student_", "teacher_"]:
            rgb_image = data_dict[img_type + "image_1"]
            label_image = data_dict[img_type + "mask_1"]

            encoded_inputs = self.feature_extractor(
                rgb_image, label_image, return_tensors="pt"
            )
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
            data_dict[img_type + "pixel_values_1"] = encoded_inputs["pixel_values"]
            data_dict[img_type + "labels_1"] = encoded_inputs["labels"]
            del data_dict[img_type + "image_1"]
            del data_dict[img_type + "mask_1"]
        data_dict["original_mask_1"] = torch.from_numpy(np.array(data_dict["original_mask_1"])).to(torch.long)
        if "instance_mask_original" in data_dict:
            data_dict["instance_mask_original"] = torch.from_numpy(
                np.array(data_dict["instance_mask_original"])
            ).clone()
        if "pseudo_instance" in data_dict:
            data_dict["pseudo_instance"] = torch.from_numpy(
                np.array(data_dict["pseudo_instance"])
            ).clone()
        return data_dict


@TRANSFORMS2D.register_module()
class RescaleAndDistort(object):
    def __init__(
        self,
        min_scale_factor=0.5,
        max_scale_factor=1.2,
        min_distort_factor=0.9,
        max_distort_factor=1.1,
        student_only=False,
        scale_original_mask=True,
    ):
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.min_distort_factor = min_distort_factor
        self.max_distort_factor = max_distort_factor
        self.student_only = student_only
        self.scale_original_mask = scale_original_mask

    def __call__(self, data_dict):
        # 1) Sample anisotropic scaling that preserves area
        rescale_factor = np.random.uniform(self.min_scale_factor, self.max_scale_factor)
        distortion_factor = np.random.uniform(
            self.min_distort_factor, self.max_distort_factor
        )
        s_x = rescale_factor * distortion_factor
        s_y = rescale_factor / distortion_factor

        # 2) Choose a reference image to derive the final target size once
        ref_key = "student_" if ("student_image_1" in data_dict) else "teacher_"
        ref_img = data_dict[ref_key + "image_1"]
        old_w, old_h = ref_img.size  # PIL: (width, height)

        # 3) Compute initial target size
        tgt_w = max(1, int(round(old_w * s_x)))
        tgt_h = max(1, int(round(old_h * s_y)))

        # 4) Enforce minimal height (keep your original intent), ensure not smaller than original
        if tgt_h < old_h:
            scale = old_h / max(1, tgt_h)
            scale_w = int(round(tgt_w * scale))
            scale_h = int(round(tgt_h * scale))
            if scale_w < old_w:
                scale_w = old_w
            if scale_h < old_h:
                scale_h = old_h
            tgt_w, tgt_h = scale_w, scale_h

        # Final effective scale (with integer rounding included)
        s_x_total = tgt_w / old_w
        s_y_total = tgt_h / old_h

        # 5) Resize student/teacher consistently
        img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
        for img_type in img_types:
            rgb_image = data_dict[img_type + "image_1"]
            label_image = data_dict[img_type + "mask_1"]
            data_dict[img_type + "image_1"] = rgb_image.resize(
                (tgt_w, tgt_h), Image.BILINEAR
            )
            data_dict[img_type + "mask_1"] = label_image.resize(
                (tgt_w, tgt_h), Image.NEAREST
            )

        # 6) Resize auxiliary maps once, consistently
        if "original_mask_1" in data_dict and self.scale_original_mask:
            data_dict["original_mask_1"] = data_dict["original_mask_1"].resize(
                (tgt_w, tgt_h), Image.NEAREST
            )

        if "instance_mask_original" in data_dict:
            data_dict["instance_mask_original"] = data_dict[
                "instance_mask_original"
            ].resize((tgt_w, tgt_h), Image.NEAREST)

        if "depth" in data_dict:
            data_dict["depth"] = data_dict["depth"].resize(
                (tgt_w, tgt_h), Image.BILINEAR
            )

        if "pseudo_instance" in data_dict:
            data_dict["pseudo_instance"] = data_dict["pseudo_instance"].resize(
                (tgt_w, tgt_h), Image.NEAREST
            )

        # 7) Update pixel coordinates (v=row=y -> scale by s_y, u=col=x -> scale by s_x)
        if "pixel_coords_array" in data_dict:
            pixel_coords_array = data_dict["pixel_coords_array"].astype(np.float32)
            # v (row) in [:,0], u (col) in [:,1]
            pixel_coords_array[:, 1] *= s_x_total
            pixel_coords_array[:, 0] *= s_y_total

            # Round to nearest integer pixel
            pixel_coords_array = np.rint(pixel_coords_array).astype(np.int32)

            # Invalidate out-of-bounds
            valid_mask = (
                (pixel_coords_array[:, 1] >= 0)
                & (pixel_coords_array[:, 1] < tgt_w)
                & (pixel_coords_array[:, 0] >= 0)
                & (pixel_coords_array[:, 0] < tgt_h)
            )
            pixel_coords_array[~valid_mask] = -1
            data_dict["pixel_coords_array"] = pixel_coords_array

        return data_dict


# Backup:
"""
    def __call__(self, data_dict):
        rescale_factor = np.random.uniform(self.min_scale_factor, self.max_scale_factor)
        distortion_factor = np.random.uniform(
            self.min_distort_factor, self.max_distort_factor
        )

        s_x = rescale_factor * distortion_factor
        s_y = rescale_factor * (1 / distortion_factor)

        img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
        for img_type in img_types:
            rgb_image = data_dict[img_type + "image_1"]
            old_size = rgb_image.size  # (width, height)

            label_image = data_dict[img_type + "mask_1"]

            # Resize images using initial scaling factors
            new_width = int(old_size[0] * s_x)
            new_height = int(old_size[1] * s_y)
            rgb_image = rgb_image.resize((new_width, new_height), Image.BILINEAR)
            label_image = label_image.resize((new_width, new_height), Image.NEAREST)
            if "original_mask_1" in data_dict and self.scale_original_mask:
                original_mask = data_dict["original_mask_1"].copy()
                original_mask = original_mask.resize(
                    (new_width, new_height), Image.NEAREST
                )
            if "instance_mask_original" in data_dict:
                instance_mask_original = data_dict["instance_mask_original"].copy()
                instance_mask_original = instance_mask_original.resize(
                    (new_width, new_height), Image.NEAREST
                )
            if "depth" in data_dict:
                depth_image = data_dict["depth"].copy()
                depth_image = depth_image.resize(
                    (new_width, new_height), Image.BILINEAR
                )
            if "pseudo_instance" in data_dict:
                pseudo_instance = data_dict["pseudo_instance"].copy()
                pseudo_instance = pseudo_instance.resize(
                    (new_width, new_height), Image.NEAREST
                )
            # Check for minimal image height
            if rgb_image.size[1] < old_size[1]:
                scale = old_size[1] / rgb_image.size[1]
                scale_x = int(rgb_image.size[0] * scale)
                scale_y = int(rgb_image.size[1] * scale)
                if scale_x < old_size[0]:
                    scale_x = old_size[0]
                if scale_y < old_size[1]:
                    scale_y = old_size[1]

                rgb_image = rgb_image.resize((scale_x, scale_y), Image.BILINEAR)
                label_image = label_image.resize((scale_x, scale_y), Image.NEAREST)
                if "depth" in data_dict:
                    depth_image = depth_image.resize((scale_x, scale_y), Image.BILINEAR)
                if "pseudo_instance" in data_dict:
                    pseudo_instance = pseudo_instance.resize(
                        (new_width, new_height), Image.NEAREST
                    )
                if "original_mask_1" in data_dict and self.scale_original_mask:
                    original_mask = original_mask.resize(
                        (new_width, new_height), Image.NEAREST
                    )
                if "instance_mask_original" in data_dict and self.scale_original_mask:
                    instance_mask_original = instance_mask_original.resize(
                        (new_width, new_height), Image.NEAREST
                    )

                additional_s_x = scale_x / new_width
                additional_s_y = scale_y / new_height
                s_x_total = s_x * additional_s_x
                s_y_total = s_y * additional_s_y
                new_width = scale_x
                new_height = scale_y
            else:
                s_x_total = s_x
                s_y_total = s_y

            data_dict[img_type + "image_1"] = rgb_image
            data_dict[img_type + "mask_1"] = label_image

        if "original_mask_1" in data_dict and self.scale_original_mask:
            data_dict["original_mask_1"] = original_mask
        if "instance_mask_original" in data_dict:
            data_dict["instance_mask_original"] = instance_mask_original
        if "depth" in data_dict:
            data_dict["depth"] = depth_image
        if "pseudo_instance" in data_dict:
            data_dict["pseudo_instance"] = pseudo_instance
        if "pixel_coords_array" in data_dict:
            pixel_coords_array = data_dict["pixel_coords_array"]
            pixel_coords_array[:, 1] = (
                pixel_coords_array[:, 1] * s_x_total
            )  # Adjust u-coordinate
            pixel_coords_array[:, 0] = (
                pixel_coords_array[:, 0] * s_y_total
            )  # Adjust v-coordinate
            pixel_coords_array = pixel_coords_array.astype(int)
            valid_mask = (
                (pixel_coords_array[:, 1] >= 0)
                & (pixel_coords_array[:, 1] < new_width)
                & (pixel_coords_array[:, 0] >= 0)
                & (pixel_coords_array[:, 0] < new_height)
            )
            pixel_coords_array[~valid_mask] = -1
            data_dict["pixel_coords_array"] = pixel_coords_array
        return data_dict"""


@TRANSFORMS2D.register_module()
class RandHorizontalFlip(object):
    def __init__(self, p=0.5, student_only=False):
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            img_types = ["student_"] if self.student_only else ["student_", "teacher_"]

            # Get image width before flipping (assuming all images have the same width)
            rgb_image = data_dict[img_types[0] + "image_1"]
            image_width = rgb_image.size[0]  # For PIL images, size[0] is the width

            # Flip images and masks
            for img_type in img_types:
                rgb_image = data_dict[img_type + "image_1"]
                label_image = data_dict[img_type + "mask_1"]

                # Flip the images
                data_dict[img_type + "image_1"] = rgb_image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
                data_dict[img_type + "mask_1"] = label_image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

            if "original_mask_1" in data_dict:
                data_dict["original_mask_1"] = data_dict["original_mask_1"].transpose(
                    Image.FLIP_LEFT_RIGHT
                )

            if "depth" in data_dict:
                depth_image = data_dict["depth"]
                data_dict["depth"] = depth_image.transpose(Image.FLIP_LEFT_RIGHT)
            if "pseudo_instance" in data_dict:
                pseudo_instance = data_dict["pseudo_instance"]
                data_dict["pseudo_instance"] = pseudo_instance.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
            if "instance_mask_original" in data_dict:
                instance_mask_original = data_dict["instance_mask_original"]
                data_dict["instance_mask_original"] = instance_mask_original.transpose(
                    Image.FLIP_LEFT_RIGHT
                )

            if "pixel_coords_array" in data_dict:
                pixel_coords_array = data_dict["pixel_coords_array"].copy()
                valid_mask = pixel_coords_array[:, 1] != -1
                pixel_coords_array[valid_mask, 1] = (
                    image_width - pixel_coords_array[valid_mask, 1] - 1
                )
                data_dict["pixel_coords_array"] = pixel_coords_array
        return data_dict


@TRANSFORMS2D.register_module()
class Mast3rCrop(object):
    def __init__(self):
        pass

    def __call__(self, data_dict):
        img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
        rgb_image = data_dict[img_types[0] + "image_1"]

        W1, H1 = rgb_image.size
        S = max(W1, H1)

        scale_factor = 512 / S

        W_scaled = round(W1 * scale_factor)
        H_scaled = round(H1 * scale_factor)

        W_cropped = (W_scaled // 16) * 16
        H_cropped = (H_scaled // 16) * 16

        W_final = round(W_cropped / scale_factor)
        H_final = round(H_cropped / scale_factor)

        cx, cy = W1 // 2, H1 // 2
        halfw, halfh = W_final // 2, H_final // 2

        for img_type in img_types:
            rgb_image = data_dict[img_type + "image_1"]
            label_image = data_dict[img_type + "mask_1"]

            rgb_image = rgb_image.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            label_image = label_image.crop(
                (cx - halfw, cy - halfh, cx + halfw, cy + halfh)
            )

            data_dict[img_type + "image_1"] = rgb_image
            data_dict[img_type + "mask_1"] = label_image
        return data_dict


@TRANSFORMS2D.register_module()
class RandCrop(object):
    def __init__(self, crop_size=(960, 960), student_only=False):
        self.crop_size = crop_size
        self.student_only = student_only

    def __call__(self, data_dict):
        img_types = ["student_"] if self.student_only else ["student_", "teacher_"]

        # Get the crop parameters using the first image
        rgb_image = data_dict[img_types[0] + "image_1"]

        # Import RandomCrop from torchvision.transforms
        i, j, h, w = transforms.RandomCrop.get_params(
            rgb_image, output_size=self.crop_size
        )

        for img_type in img_types:
            rgb_image = data_dict[img_type + "image_1"]
            label_image = data_dict[img_type + "mask_1"]

            # Apply the same crop to both image and mask
            rgb_image = TF.crop(rgb_image, i, j, h, w)
            label_image = TF.crop(label_image, i, j, h, w)

            data_dict[img_type + "image_1"] = rgb_image
            data_dict[img_type + "mask_1"] = label_image

        if "original_mask_1" in data_dict:
            data_dict["original_mask_1"] = TF.crop(
                data_dict["original_mask_1"], i, j, h, w
            )

        if "depth" in data_dict:
            data_dict["depth"] = TF.crop(data_dict["depth"], i, j, h, w)
        if "pseudo_instance" in data_dict:
            data_dict["pseudo_instance"] = TF.crop(
                data_dict["pseudo_instance"], i, j, h, w
            )
        if "instance_mask_original" in data_dict:
            data_dict["instance_mask_original"] = TF.crop(
                data_dict["instance_mask_original"], i, j, h, w
            )

        if "pixel_coords_array" in data_dict:
            pixel_coords_array = data_dict["pixel_coords_array"].copy()
            pixel_coords_array[:, 0] -= i  # v (row)
            pixel_coords_array[:, 1] -= j  # u (col)
            valid_mask = (
                (pixel_coords_array[:, 0] >= 0)
                & (pixel_coords_array[:, 0] < h)
                & (pixel_coords_array[:, 1] >= 0)
                & (pixel_coords_array[:, 1] < w)
            )
            pixel_coords_array[~valid_mask] = -1
            data_dict["pixel_coords_array"] = pixel_coords_array

        return data_dict


@TRANSFORMS2D.register_module()
class Jitter(object):
    def __init__(
        self, brightness=0.25, contrast=0.25, saturation=0.25, student_only=False
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.student_only = student_only

    def __call__(self, data_dict):
        img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
        for img_type in img_types:
            rgb_image = data_dict[img_type + "image_1"]
            data_dict[img_type + "image_1"] = ColorJitter(
                self.brightness, self.contrast, self.saturation
            )(rgb_image)
        return data_dict


@TRANSFORMS2D.register_module()
class Blur(object):
    def __init__(self, kernel_size=7, sigma=(0.1, 1.5), p=0.5, student_only=False):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
            for img_type in img_types:
                rgb_image = data_dict[img_type + "image_1"]
                data_dict[img_type + "image_1"] = GaussianBlur(
                    self.kernel_size, self.sigma
                )(rgb_image)
        return data_dict


@TRANSFORMS2D.register_module()
class AugMix(object):
    def __init__(self, severity=2, p=0.5, student_only=False):
        self.severity = severity
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
            for img_type in img_types:
                rgb_image = data_dict[img_type + "image_1"]
                data_dict[img_type + "image_1"] = torchvision.transforms.AugMix(
                    severity=self.severity
                )(rgb_image)
        return data_dict


@TRANSFORMS2D.register_module()
class Cutout(object):
    def __init__(self, square_size=90, p=0.5, student_only=False):
        self.square_size = square_size
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
            for img_type in img_types:
                rgb_image = data_dict[img_type + "image_1"]
                sqare_size = self.square_size
                x1 = np.random.randint(0, rgb_image.size[0]) - sqare_size // 2
                y1 = np.random.randint(0, rgb_image.size[1]) - sqare_size // 2
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x1 + sqare_size > rgb_image.size[0]:
                    x1 = rgb_image.size[0] - sqare_size
                if y1 + sqare_size > rgb_image.size[1]:
                    y1 = rgb_image.size[1] - sqare_size
                rgb_image = np.array(rgb_image)
                rgb_image[x1 : x1 + sqare_size, y1 : y1 + sqare_size, :] = (0, 0, 0)
                data_dict[img_type + "image_1"] = Image.fromarray(rgb_image)
        return data_dict


@TRANSFORMS2D.register_module()
class VShift(object):
    def __init__(self, min_shift=-50, max_shift=50, p=0.5, student_only=False):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            img_types = ["student_"] if self.student_only else ["student_", "teacher_"]
            for img_type in img_types:
                rgb_image = data_dict[img_type + "image_1"]
                label_image = data_dict[img_type + "mask_1"]
                shift = np.random.randint(self.min_shift, self.max_shift)
                if shift == 0:
                    # Zero shift is problematic
                    shift = 1
                rgb_image = np.array(rgb_image)
                label_image = np.array(label_image)
                img_size = rgb_image.shape
                if shift > 0:
                    rgb_image = rgb_image[shift:, :, :]
                    label_image = label_image[shift:, :]
                else:
                    rgb_image = rgb_image[:shift, :, :]
                    label_image = label_image[:shift, :]
                # Resize back to old size
                try:
                    label_image = Image.fromarray(label_image)
                    rgb_image = Image.fromarray(rgb_image)
                    label_image = label_image.resize(
                        (img_size[1], img_size[0]), Image.NEAREST
                    )
                    rgb_image = rgb_image.resize(
                        (img_size[1], img_size[0]), Image.BILINEAR
                    )
                except:
                    print("Error: Could not resize the image")
                    return None, None
                data_dict[img_type + "image_1"] = rgb_image
                data_dict[img_type + "mask_1"] = label_image
        return data_dict


class Compose2D(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS2D.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
