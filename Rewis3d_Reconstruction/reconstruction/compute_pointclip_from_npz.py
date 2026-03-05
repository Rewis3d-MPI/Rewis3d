#!/usr/bin/env python3
# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Utility: Suggest a PointClip range from NPZ files produced by the reconstruction pipeline.

It randomly samples up to N files from a folder (recursively), aggregates the 3D points
from data_3d['student_coord'], computes robust min/max using percentiles to ignore outliers,
and prints a config line like:

  dict(type="PointClip", point_cloud_range=(xmin, ymin, zmin, xmax, ymax, zmax)),

It also prints an optional symmetric alternative (per-axis) and basic stats.

Example:
  python compute_pointclip_from_npz.py \
      --root /path/to/npz_dir \
      --samples 50 \
      --low 1.0 --high 99.0 \
      --round 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np


def find_npz_files(root: str) -> list[str]:
    npz_files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".npz"):
                npz_files.append(os.path.join(dirpath, fn))
    return npz_files


def load_coords_from_npz(path: str) -> np.ndarray | None:
    """Load 3D coordinates array from a single NPZ file.

    Expects npz['data_3d'] to be a dict (saved via allow_pickle) with a key like
    'student_coord' -> (N, 3). Returns None if unavailable.
    """
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "data_3d" not in npz:
                return None
            data_3d = npz["data_3d"].item() if np.isscalar(npz["data_3d"]) or npz["data_3d"].dtype == object else None
            if data_3d is None:
                # Fallback: sometimes saved directly as dict-like object array
                try:
                    data_3d = npz["data_3d"].tolist()
                except Exception:
                    return None

            # Try common coordinate keys in order of likelihood
            for key in ("student_coord", "points", "coords", "xyz"):
                coords = data_3d.get(key)
                if coords is not None:
                    coords = np.asarray(coords)
                    if coords.ndim == 2 and coords.shape[1] >= 3:
                        # Use first 3 columns as x,y,z
                        coords = coords[:, :3]
                        # Filter invalid/NaN/Inf
                        mask = np.all(np.isfinite(coords), axis=1)
                        return coords[mask]
            return None
    except Exception:
        return None


def robust_range(coords_stack: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute robust per-axis min/max using percentiles (ignores extreme outliers)."""
    mins = np.percentile(coords_stack, low, axis=0)
    maxs = np.percentile(coords_stack, high, axis=0)
    return mins, maxs


def symmetric_range(mins: np.ndarray, maxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.maximum(np.abs(mins), np.abs(maxs))
    return -rad, rad


def round_bounds(mins: np.ndarray, maxs: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    if step <= 0:
        return mins, maxs
    mins_rounded = np.floor(mins / step) * step
    maxs_rounded = np.ceil(maxs / step) * step
    return mins_rounded, maxs_rounded


def format_pointclip(mins: np.ndarray, maxs: np.ndarray) -> str:
    vals = (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
    # Keep nice compact formatting
    def f(v):
        # Avoid trailing .0 if integer-like
        return int(v) if abs(v - int(v)) < 1e-9 else round(float(v), 4)
    vals_fmt = tuple(f(v) for v in vals)
    return f'dict(type="PointClip", point_cloud_range=({vals_fmt[0]}, {vals_fmt[1]}, {vals_fmt[2]}, {vals_fmt[3]}, {vals_fmt[4]}, {vals_fmt[5]})),'


def main():
    parser = argparse.ArgumentParser(description="Suggest PointClip range from NPZ files")
    parser.add_argument("--root", required=True, help="Root folder to search for NPZ files (recursively)")
    parser.add_argument("--samples", type=int, default=50, help="Max number of NPZ files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--low", type=float, default=1.0, help="Low percentile (e.g., 1.0)")
    parser.add_argument("--high", type=float, default=99.0, help="High percentile (e.g., 99.0)")
    parser.add_argument("--round", dest="round_step", type=float, default=1.0, help="Round step for bounds (e.g., 1.0 or 5.0)")
    parser.add_argument("--symmetric", action="store_true", help="Force symmetric bounds per axis (useful for some datasets)")
    args = parser.parse_args()

    npz_files = find_npz_files(args.root)
    if not npz_files:
        print(f"No NPZ files found under: {args.root}", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    files_sampled = random.sample(npz_files, min(args.samples, len(npz_files)))

    coords_all = []
    skipped = 0
    for p in files_sampled:
        coords = load_coords_from_npz(p)
        if coords is None or coords.size == 0:
            skipped += 1
            continue
        coords_all.append(coords)

    if not coords_all:
        print("No coordinates found in sampled NPZ files. Check that data_3d['student_coord'] exists.", file=sys.stderr)
        sys.exit(2)

    # Concatenate a capped number of points per file to avoid excessive memory
    # Take up to 100k points per file for statistics
    capped = []
    per_file_cap = 100_000
    for c in coords_all:
        if c.shape[0] > per_file_cap:
            idx = np.random.default_rng(args.seed).choice(c.shape[0], size=per_file_cap, replace=False)
            capped.append(c[idx])
        else:
            capped.append(c)
    coords_stack = np.concatenate(capped, axis=0)

    # Compute robust bounds
    mins, maxs = robust_range(coords_stack, args.low, args.high)
    if args.symmetric:
        mins, maxs = symmetric_range(mins, maxs)

    # Rounded suggestions
    mins_r, maxs_r = round_bounds(mins, maxs, args.round_step)

    # Also provide a symmetric rounded suggestion as an alternative
    s_mins, s_maxs = symmetric_range(mins, maxs)
    s_mins_r, s_maxs_r = round_bounds(s_mins, s_maxs, args.round_step)

    # Print summary
    print("Files scanned:", len(files_sampled), f"(skipped {skipped} without coords)")
    means = coords_stack.mean(axis=0)
    stds = coords_stack.std(axis=0)
    raw_min = coords_stack.min(axis=0)
    raw_max = coords_stack.max(axis=0)
    print(f"Per-axis mean: {means}")
    print(f"Per-axis std:  {stds}")
    print(f"Raw min:       {raw_min}")
    print(f"Raw max:       {raw_max}")
    print(f"Percentile {args.low:.2f}-{args.high:.2f} mins/maxs: {mins} / {maxs}")

    print("\nRecommended (rounded):")
    print("  ", format_pointclip(mins_r, maxs_r))

    print("\nSymmetric alternative (rounded):")
    print("  ", format_pointclip(s_mins_r, s_maxs_r))

    print("\nTip: For outdoor datasets, symmetric ranges are often preferred; for indoor datasets, non-symmetric can be tighter.")


if __name__ == "__main__":
    main()
