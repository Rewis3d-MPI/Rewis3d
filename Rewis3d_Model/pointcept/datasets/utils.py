# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence

from pointcept.utils.logger import get_root_logger


def collate_fn(batch):
    """
    Custom collate function to handle both 3D point cloud data and 2D image data, including variable-length sequences.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{type(batch)} is not supported.")

    elem = batch[0]

    if isinstance(elem, Mapping):
        collated_batch = {}
        for key in elem:
            values = [d[key] for d in batch]

            if key in [
                "student_coord",
                "student_grid_coord",
                "student_feat",
                "student_segment",
                "student_inverse",
                "student_colors",
                "student_normal",
                "teacher_coord",
                "teacher_grid_coord",
                "teacher_feat",
                "teacher_segment",
                "teacher_inverse",
                "teacher_colors",
                "teacher_normal",
                "student_sampled_indices",
                "teacher_sampled_indices",
                "original_segment",
                "correspondences",
                "conf",
            ]:
                # Point cloud data: concatenate along the first dimension
                collated_batch[key] = torch.cat(values, dim=0)

            elif key in ["student_offset", "teacher_offset", "original_offset"]:
                # 'offset' indicates the number of points in each sample
                lengths = [
                    v.item() if isinstance(v, torch.Tensor) else v for v in values
                ]
                offsets = torch.tensor(lengths, dtype=torch.long)
                collated_batch[key] = torch.cumsum(offsets, dim=0)

            elif key in ["student_pixel_values_1", "teacher_pixel_values_1"]:
                # Image data: stack along the batch dimension
                collated_batch[key] = torch.stack(values, dim=0)

            elif key in ["student_labels_1", "teacher_labels_1", "original_mask_1"]:
                # Labels or masks for images: stack along the batch dimension
                collated_batch[key] = torch.stack(values, dim=0)

            elif key in ["point_indices_array", "pixel_coords_array"]:
                # Handle variable-length arrays by padding
                max_length = max(v.shape[0] for v in values)
                padded_values = []
                masks = []
                for v in values:
                    length = v.shape[0]
                    pad_size = max_length - length
                    # Pad the array
                    if key == "point_indices_array":
                        padded_v = torch.cat(
                            [
                                v,
                                torch.full(
                                    (pad_size,), -1, dtype=v.dtype, device=v.device
                                ),
                            ]
                        )
                    elif key == "pixel_coords_array":
                        padded_v = torch.cat(
                            [
                                v,
                                torch.full(
                                    (pad_size, v.shape[1]),
                                    -1,
                                    dtype=v.dtype,
                                    device=v.device,
                                ),
                            ]
                        )
                    padded_values.append(padded_v)
                    # Create mask
                    mask = torch.cat(
                        [
                            torch.ones(length, dtype=torch.bool, device=v.device),
                            torch.zeros(pad_size, dtype=torch.bool, device=v.device),
                        ]
                    )
                    masks.append(mask)
                # Stack the padded arrays and masks
                collated_batch[key] = torch.stack(padded_values, dim=0)
                collated_batch[key + "_mask"] = torch.stack(masks, dim=0)

            else:
                # For any other key, use the default collate
                collated_batch[key] = default_collate(values)
        return collated_batch

    elif isinstance(elem, torch.Tensor):
        # Handle tensors that can be stacked
        sizes = [tensor.size() for tensor in batch]
        if all(size == sizes[0] for size in sizes):
            return torch.stack(batch, dim=0)
        else:
            # For tensors of variable size, concatenate along the first dimension
            return torch.cat(batch, dim=0)

    elif isinstance(elem, Sequence) and not isinstance(elem, str):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    else:
        return default_collate(batch)


def fcollate_fn(batch):
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    # if "offset" in batch.keys():
    #    # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
    #    if random.random() < mix_prob:
    #        batch["offset"] = torch.cat(
    #            [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
    #        )
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
