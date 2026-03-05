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

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(
                student_offset="student_coord", teacher_offset="teacher_coord"
            )
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
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


@TRANSFORMS.register_module()
class AddZeroNormal(object):
    """Add zero normals for point clouds that don't have normal information.

    This is useful for using pretrained models (like Concerto) that expect
    9-channel input (coord + color + normal) when only coord + color are available.
    """

    def __init__(self, coord_key="student_coord", normal_key="student_normal"):
        self.coord_key = coord_key
        self.normal_key = normal_key

    def __call__(self, data_dict):
        coord = data_dict[self.coord_key]
        if isinstance(coord, np.ndarray):
            data_dict[self.normal_key] = np.zeros_like(coord)
        elif isinstance(coord, torch.Tensor):
            data_dict[self.normal_key] = torch.zeros_like(coord)
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
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


@TRANSFORMS.register_module()
class MapIds(object):
    def __init__(self, id2trainId_segment, id2trainId_original_segment=None):
        self.id2trainId_segment = id2trainId_segment
        self.id2trainId_original_segment = id2trainId_original_segment

    def __call__(self, data_dict):
        if self.id2trainId_original_segment is not None:
            segment = np.array(data_dict["original_segment"])
            mapped_labels = np.copy(segment)
            for original_value, new_value in self.id2trainId_original_segment.items():
                mapped_labels[segment == original_value] = new_value
            data_dict["original_segment"] = mapped_labels

        if self.id2trainId_segment is not None:
            for mode in ["student_"]:
                # , "teacher_"]:
                segment = np.array(data_dict[mode + "segment"])
                mapped_labels = np.copy(segment)
                for original_value, new_value in self.id2trainId_segment.items():
                    mapped_labels[segment == original_value] = new_value
                data_dict[mode + "segment"] = mapped_labels
        return data_dict


@TRANSFORMS.register_module()
class CenterToOrigin(object):
    def __call__(self, data_dict):
        for mode in ["student_", "teacher_"]:
            key = mode + "coord"
            if key in data_dict:
                centroid = np.mean(data_dict[key], axis=0)
                data_dict[key] = data_dict[key] - centroid
        return data_dict


@TRANSFORMS.register_module()
class Add(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            for mode in ["student_", "teacher_"]:
                # modified from pointnet2
                centroid = np.mean(data_dict[mode + "coord"], axis=0)
                data_dict[mode + "coord"] -= centroid
                m = np.max(np.sqrt(np.sum(data_dict[mode + "coord"] ** 2, axis=1)))
                data_dict[mode + "coord"] = data_dict[mode + "coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            for mode in ["student_", "teacher_"]:
                coord_min = np.min(data_dict[mode + "coord"], 0)
                data_dict[mode + "coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            for mode in ["student_", "teacher_"]:
                x_min, y_min, z_min = data_dict[mode + "coord"].min(axis=0)
                x_max, y_max, _ = data_dict[mode + "coord"].max(axis=0)
                if self.apply_z:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
                else:
                    shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
                data_dict[mode + "coord"] -= shift

        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0)), student_only=False):
        self.shift = shift
        self.student_only = student_only

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["student_coord"] += [shift_x, shift_y, shift_z]
            if not self.student_only:
                data_dict["teacher_coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1), student_only=False):
        self.point_cloud_range = point_cloud_range
        self.student_only = student_only

    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            data_dict["student_coord"] = np.clip(
                data_dict["student_coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
            if not self.student_only:
                data_dict["teacher_coord"] = np.clip(
                    data_dict["teacher_coord"],
                    a_min=self.point_cloud_range[:3],
                    a_max=self.point_cloud_range[3:],
                )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(
        self, dropout_ratio=0.2, dropout_application_ratio=0.5, student_only=False
    ):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio
        self.student_only = student_only

    def __call__(self, data_dict):
        modes = ["student_"] if self.student_only else ["student_", "teacher_"]
        for mode in modes:
            if random.random() < self.dropout_application_ratio:
                n = len(data_dict[mode + "coord"])
                idx = np.random.choice(
                    n, int(n * (1 - self.dropout_ratio)), replace=False
                )
                if mode + "sampled_index" in data_dict:
                    # for ScanNet data efficient, we need to make sure labeled point is sampled.
                    idx = np.unique(np.append(idx, data_dict[mode + "sampled_index"]))
                    mask = np.zeros_like(data_dict[mode + "segment"]).astype(bool)
                    mask[data_dict[mode + "sampled_index"]] = True
                    data_dict[mode + "sampled_index"] = np.where(mask[idx])[0]
                if mode + "coord" in data_dict.keys():
                    data_dict[mode + "coord"] = data_dict[mode + "coord"][idx]
                if mode + "color" in data_dict.keys():
                    data_dict[mode + "color"] = data_dict[mode + "color"][idx]
                if mode + "normal" in data_dict.keys():
                    data_dict[mode + "normal"] = data_dict[mode + "normal"][idx]
                if mode + "strength" in data_dict.keys():
                    data_dict[mode + "strength"] = data_dict[mode + "strength"][idx]
                if mode + "segment" in data_dict.keys():
                    data_dict[mode + "segment"] = data_dict[mode + "segment"][idx]
                if mode + "instance" in data_dict.keys():
                    data_dict[mode + "instance"] = data_dict[mode + "instance"][idx]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(
        self,
        angle=None,
        center=None,
        axis="z",
        always_apply=False,
        p=0.5,
        student_only=False,
    ):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center
        self.student_only = student_only

    def __call__(self, data_dict):
        modes = ["student_"] if self.student_only else ["student_", "teacher_"]
        for mode in modes:
            if random.random() < self.p:
                angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
                rot_cos, rot_sin = np.cos(angle), np.sin(angle)
                if self.axis == "x":
                    rot_t = np.array(
                        [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]]
                    )
                elif self.axis == "y":
                    rot_t = np.array(
                        [[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]]
                    )
                elif self.axis == "z":
                    rot_t = np.array(
                        [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                    )
                else:
                    raise NotImplementedError
                if mode + "coord" in data_dict.keys():
                    if self.center is None:
                        x_min, y_min, z_min = data_dict[mode + "coord"].min(axis=0)
                        x_max, y_max, z_max = data_dict[mode + "coord"].max(axis=0)
                        center = [
                            (x_min + x_max) / 2,
                            (y_min + y_max) / 2,
                            (z_min + z_max) / 2,
                        ]
                    else:
                        center = self.center
                    data_dict[mode + "coord"] -= center
                    data_dict[mode + "coord"] = np.dot(
                        data_dict[mode + "coord"], np.transpose(rot_t)
                    )
                    data_dict[mode + "coord"] += center
                if mode + "normal" in data_dict.keys():
                    data_dict[mode + "normal"] = np.dot(
                        data_dict[mode + "normal"], np.transpose(rot_t)
                    )
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self,
        angle=(1 / 2, 1, 3 / 2),
        center=None,
        axis="z",
        always_apply=False,
        p=0.75,
        student_only=False,
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center
        self.student_only = student_only

    def __call__(self, data_dict):
        modes = ["student_"] if self.student_only else ["student_", "teacher_"]
        for mode in modes:
            if random.random() < self.p:
                angle = np.random.choice(self.angle) * np.pi
                rot_cos, rot_sin = np.cos(angle), np.sin(angle)
                if self.axis == "x":
                    rot_t = np.array(
                        [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]]
                    )
                elif self.axis == "y":
                    rot_t = np.array(
                        [[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]]
                    )
                elif self.axis == "z":
                    rot_t = np.array(
                        [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                    )
                else:
                    raise NotImplementedError
                if mode + "coord" in data_dict.keys():
                    if self.center is None:
                        x_min, y_min, z_min = data_dict[mode + "coord"].min(axis=0)
                        x_max, y_max, z_max = data_dict[mode + "coord"].max(axis=0)
                        center = [
                            (x_min + x_max) / 2,
                            (y_min + y_max) / 2,
                            (z_min + z_max) / 2,
                        ]
                    else:
                        center = self.center
                    data_dict[mode + "coord"] -= center
                    data_dict[mode + "coord"] = np.dot(
                        data_dict[mode + "coord"], np.transpose(rot_t)
                    )
                    data_dict[mode + "coord"] += center
                if mode + "normal" in data_dict.keys():
                    data_dict[mode + "normal"] = np.dot(
                        data_dict[mode + "normal"], np.transpose(rot_t)
                    )
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False, student_only=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic
        self.student_only = student_only

    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["student_coord"] *= scale

            if not self.student_only:
                scale = np.random.uniform(
                    self.scale[0], self.scale[1], 3 if self.anisotropic else 1
                )
                data_dict["teacher_coord"] *= scale

        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5, student_only=False):
        self.p = p
        self.student_only = student_only

    def __call__(self, data_dict):
        modes = ["student_"] if self.student_only else ["student_", "teacher_"]
        for mode in modes:
            if np.random.rand() < self.p:
                if mode + "coord" in data_dict.keys():
                    data_dict[mode + "coord"][:, 0] = -data_dict[mode + "coord"][:, 0]
                if mode + "normal" in data_dict.keys():
                    data_dict[mode + "normal"][:, 0] = -data_dict[mode + "normal"][:, 0]
            if np.random.rand() < self.p:
                if mode + "coord" in data_dict.keys():
                    data_dict[mode + "coord"][:, 1] = -data_dict[mode + "coord"][:, 1]
                if mode + "normal" in data_dict.keys():
                    data_dict[mode + "normal"][:, 1] = -data_dict[mode + "normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05, student_only=False):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip
        self.student_only = student_only

    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["student_coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["student_coord"] += jitter

            if not self.student_only:
                jitter = np.clip(
                    self.sigma
                    * np.random.randn(data_dict["teacher_coord"].shape[0], 3),
                    -self.clip,
                    self.clip,
                )
                data_dict["teacher_coord"] += jitter

        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False, student_only=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter
        self.student_only = student_only

    def __call__(self, data_dict):
        if "student_coord" in data_dict.keys():
            modes = ["student_"] if self.student_only else ["student_", "teacher_"]
            for mode in modes:
                jitter = np.random.multivariate_normal(
                    self.mean, self.cov, data_dict[mode + "coord"].shape[0]
                )
                jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
                data_dict[mode + "coord"] += jitter
                if self.store_jitter:
                    data_dict[mode + "jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None, student_only=False):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )
        self.student_only = student_only

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        modes = ["student_"] if self.student_only else ["student_", "teacher_"]
        for mode in modes:
            if (
                mode + "coord" in data_dict.keys()
                and self.distortion_params is not None
            ):
                if random.random() < 0.95:
                    for granularity, magnitude in self.distortion_params:
                        data_dict[mode + "coord"] = self.elastic_distortion(
                            data_dict[mode + "coord"], granularity, magnitude
                        )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        key_prefix="",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        return_sampled_indices=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.key_prefix = key_prefix
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.return_sampled_indices = return_sampled_indices
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        prefix = self.key_prefix
        assert prefix + "coord" in data_dict.keys()
        scaled_coord = data_dict[prefix + "coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # Ensure labeled points are sampled
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["student_segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_inverse:
                data_dict[prefix + "inverse"] = np.zeros_like(inverse)
                data_dict[prefix + "inverse"][idx_sort] = inverse
            if self.return_sampled_indices:
                data_dict[prefix + "sampled_indices"] = idx_unique
            if self.return_grid_coord:
                data_dict[prefix + "grid_coord"] = grid_coord[idx_unique]
            if self.return_min_coord:
                data_dict[prefix + "min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict[prefix + "student_normal"],
                        axis=-1,
                        keepdims=True,
                    )
                data_dict["displacement"] = displacement[idx_unique]
            # Apply idx_unique to student and teacher keys
            for key in self.keys:
                full_key = prefix + key
                if full_key in data_dict:
                    data_dict[full_key] = data_dict[full_key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_inverse:
                    data_dict["inverse"] = np.zeros_like(inverse)
                    data_dict["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["student_normal"],
                            axis=-1,
                            keepdims=True,
                        )
                    data_part["displacement"] = displacement[idx_part]
                # Apply idx_part to student and teacher keys
                for key in self.keys:
                    student_key = "student_" + key
                    if student_key in data_dict:
                        data_part[student_key] = data_dict[student_key][idx_part]
                    if not self.student_only:
                        teacher_key = "teacher_" + key
                        if teacher_key in data_dict:
                            data_part[teacher_key] = data_dict[teacher_key][idx_part]
                # Copy over other keys without modification
                for key in data_dict.keys():
                    if (
                        key not in self.keys
                        and key not in ["student_" + k for k in self.keys]
                        and key not in ["teacher_" + k for k in self.keys]
                    ):
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(
        self, point_max=80000, sample_rate=None, mode="random", student_only=False
    ):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode
        self.student_only = student_only

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["student_coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "student_coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["student_coord"].shape[0])
            data_part_list = []
            if data_dict["student_coord"].shape[0] > point_max:
                coord_p, idx_uni = (
                    np.random.rand(data_dict["student_coord"].shape[0]) * 1e-3,
                    np.array([]),
                )
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(
                        np.power(
                            data_dict["student_coord"]
                            - data_dict["student_coord"][init_idx],
                            2,
                        ),
                        axis=1,
                    )
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    # Process student keys
                    for key in data_dict.keys():
                        if key.startswith("student_"):
                            data_crop_dict[key] = data_dict[key][idx_crop]
                    # Process teacher keys if not student_only
                    if not self.student_only:
                        for key in data_dict.keys():
                            if key.startswith("teacher_"):
                                data_crop_dict[key] = data_dict[key][idx_crop]
                    # Other keys
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(
                        1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"])
                    )
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(
                        np.concatenate((idx_uni, data_crop_dict["index"]))
                    )
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["student_coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # Mode is "random" or "center"
        elif data_dict["student_coord"].shape[0] > point_max:
            if self.mode == "random":
                center_idx = np.random.randint(data_dict["student_coord"].shape[0])
                center = data_dict["student_coord"][center_idx]
            elif self.mode == "center":
                center = data_dict["student_coord"][
                    data_dict["student_coord"].shape[0] // 2
                ]
            else:
                raise NotImplementedError
            dist2 = np.sum(np.square(data_dict["student_coord"] - center), axis=1)
            idx_crop = np.argsort(dist2)[:point_max]
            # Process student keys
            for key in data_dict.keys():
                if key.startswith("student_"):
                    data_dict[key] = data_dict[key][idx_crop]
            # Process teacher keys if not student_only
            if not self.student_only:
                for key in data_dict.keys():
                    if key.startswith("teacher_"):
                        data_dict[key] = data_dict[key][idx_crop]
            # Process other keys
            if "index" in data_dict:
                data_dict["index"] = data_dict["index"][idx_crop]
            return data_dict
        else:
            return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "student_coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)

        for mode in ["student_", "teacher_"]:
            if mode + "coord" in data_dict.keys():
                data_dict[mode + "coord"] = data_dict[mode + "coord"][shuffle_index]
            if mode + "grid_coord" in data_dict.keys():
                data_dict[mode + "grid_coord"] = data_dict[mode + "grid_coord"][
                    shuffle_index
                ]
            if mode + "displacement" in data_dict.keys():
                data_dict[mode + "displacement"] = data_dict[mode + "displacement"][
                    shuffle_index
                ]
            if mode + "color" in data_dict.keys():
                data_dict[mode + "color"] = data_dict[mode + "color"][shuffle_index]
            if mode + "normal" in data_dict.keys():
                data_dict[mode + "normal"] = data_dict[mode + "normal"][shuffle_index]
            if mode + "segment" in data_dict.keys():
                data_dict[mode + "segment"] = data_dict[mode + "segment"][shuffle_index]
            if mode + "instance" in data_dict.keys():
                data_dict["instance"] = data_dict[mode + "instance"][shuffle_index]
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        for mode in ["student_", "teacher_"]:
            if mode + "coord" in data_dict.keys():
                data_dict[mode + "coord"] = data_dict[mode + "coord"][mask]
            if mode + "grid_coord" in data_dict.keys():
                data_dict[mode + "grid_coord"] = data_dict[mode + "grid_coord"][mask]
            if mode + "color" in data_dict.keys():
                data_dict[mode + "color"] = data_dict[mode + "color"][mask]
            if mode + "normal" in data_dict.keys():
                data_dict[mode + "normal"] = data_dict[mode + "normal"][mask]
            if mode + "segment" in data_dict.keys():
                data_dict[mode + "segment"] = data_dict[mode + "segment"][mask]
            if mode + "instance" in data_dict.keys():
                data_dict[mode + "instance"] = data_dict[mode + "instance"][mask]
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
