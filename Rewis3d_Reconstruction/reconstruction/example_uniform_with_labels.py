#!/usr/bin/env python3
# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

"""
Quick example demonstrating uniform sampling with label prioritization.

Scenario: 1200 images, 8 labeled, sample 200
Result: All 8 labeled + 192 uniformly sampled unlabeled = 200 total
"""

# Example config.yaml
example_config = """
dataset:
  dataset_dir: "/path/to/dataset"
  splits: ["training"]
  
sampling:
  strategy: "uniform"        # Use uniform sampling
  num_images: 200            # Sample 200 images
  keep_all_labeled: true     # Keep ALL labeled images first
  
  # Patterns to detect labeled images
  label_detection_patterns:
    - "mask_*"
    - "label_*"
    - "scribble_*"
"""

print("=" * 70)
print("Uniform Sampling with Label Prioritization - Quick Example")
print("=" * 70)

print("\nScenario:")
print("  • 1200 total images in folder")
print("  • 8 images have labels/masks")
print("  • Want to sample 200 images")

print("\nWithout keep_all_labeled:")
print("  • Standard uniform sampling")
print("  • Every 6th image (1200/200 = 6)")
print("  • Result: ~2 labeled images (by chance)")

print("\nWith keep_all_labeled: true")
print("  • All 8 labeled images included first")
print("  • Remaining 192 slots filled uniformly from 1192 unlabeled")
print("  • Every ~6.2th unlabeled image (1192/192 ≈ 6.2)")
print("  • Result: 8 labeled + 192 unlabeled = 200 total")

print("\nConfig:")
print(example_config)

print("\nUsage:")
print("  python reconstruction/generate_dataset.py --config your_config.yaml")

print("\nTo test:")
print("  python reconstruction/test_uniform_sampling_with_labels.py")

print("\n" + "=" * 70)
