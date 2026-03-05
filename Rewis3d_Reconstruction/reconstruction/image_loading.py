import glob
import os
import re
from collections import defaultdict


def sort_function(filepath):
    """
    Robust sort key for image file paths.

    Works across datasets where the frame folder may contain prefixes, e.g.,
    'Image_0007416' or just '0007416'. It extracts the last integer sequence
    from the frame folder name for numeric sorting. Falls back to lexicographic
    order if no digits are found.
    """
    # Normalize separators and split
    parts = filepath.replace("\\", "/").split("/")
    drive_folder = parts[-3] if len(parts) >= 3 else ""
    frame_folder = parts[-2] if len(parts) >= 2 else ""

    # Extract the last integer from the frame folder (e.g., 'Image_0007416' -> 7416)
    digits = re.findall(r"\d+", frame_folder)
    frame_num = int(digits[-1]) if digits else None

    # Return a tuple that first groups by drive, then by numeric frame index when available,
    # then by frame name as a stable fallback, and finally by full path to keep sort stable.
    return (
        drive_folder,
        frame_num if frame_num is not None else float("inf"),
        frame_folder,
        filepath,
    )


# Define strategy functions
def chunk_sampling(images, n):
    """
    Divide images into chunks with maximum size n, distributing equally.
    For example: 500 images with n=200 -> 3 chunks of ~166 images each.
    """
    if len(images) <= n:
        return [images]

    # Calculate optimal number of chunks to keep each chunk <= n
    num_chunks = (len(images) + n - 1) // n  # Ceiling division
    chunk_size = len(images) // num_chunks
    remainder = len(images) % num_chunks

    chunks = []
    start_idx = 0

    for i in range(num_chunks):
        # Distribute remainder across first few chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(images[start_idx:end_idx])
        start_idx = end_idx

    return chunks


def random_sampling(images, n):
    import random

    return random.sample(images, min(n, len(images)))


def uniform_sampling(images, n, keep_all_labeled=False, label_detection_patterns=None):
    """
    Uniformly sample n images from the list.

    Args:
        images (list): List of image paths
        n (int): Target number of images to sample
        keep_all_labeled (bool): If True, prioritize keeping all labeled images first
        label_detection_patterns (list): Patterns to detect labeled images (e.g., ["mask_*", "label_*"])

    Returns:
        list: Sampled images
    """
    import glob
    import os

    import numpy as np

    if len(images) <= n:
        return images

    # If not keeping labeled images, use standard uniform sampling
    if not keep_all_labeled or label_detection_patterns is None:
        idxs = np.linspace(0, len(images) - 1, n, dtype=int)
        return [images[i] for i in idxs]

    # Detect labeled images
    labeled_indices = []
    unlabeled_indices = []

    for idx, img_path in enumerate(images):
        img_dir = os.path.dirname(img_path)
        has_label = False
        for pattern in label_detection_patterns:
            found = glob.glob(os.path.join(img_dir, pattern))
            if found:
                has_label = True
                break

        if has_label:
            labeled_indices.append(idx)
        else:
            unlabeled_indices.append(idx)

    # Start with all labeled images
    selected_indices = labeled_indices.copy()
    num_labeled = len(labeled_indices)

    # Calculate how many unlabeled images we need
    num_unlabeled_needed = n - num_labeled

    if num_unlabeled_needed <= 0:
        # Already have enough labeled images, return first n of them
        return [images[i] for i in selected_indices[:n]]

    # Uniformly sample from unlabeled images
    if len(unlabeled_indices) <= num_unlabeled_needed:
        # Keep all unlabeled images
        selected_indices.extend(unlabeled_indices)
    else:
        # Uniform sampling from unlabeled images
        uniform_idxs = np.linspace(
            0, len(unlabeled_indices) - 1, num_unlabeled_needed, dtype=int
        )
        selected_unlabeled = [unlabeled_indices[i] for i in uniform_idxs]
        selected_indices.extend(selected_unlabeled)

    # Sort indices to maintain temporal order
    selected_indices.sort()

    return [images[i] for i in selected_indices]


def uniform_sampling_grouped_keep_labels(
    images, n, keep_all_labeled, label_detection_patterns
):
    """Group-aware uniform sampling with keep-all-labeled semantics.

    Operates at the frame-folder level (e.g., Image_0007416), so we never split a frame
    across selection. All labeled frames are included, then we uniformly sample from
    unlabeled frames to fill up to n total images. Returns a single chunk (list of image paths).

    Args:
        images (List[str]): Sorted image paths for a drive/room.
        n (int): Target number of images.
        keep_all_labeled (bool): If true, always include frames that contain mask/label files.
        label_detection_patterns (List[str]): Glob patterns to detect labels in a frame folder.

    Returns:
        List[str]: Selected images as a single chunk.
    """
    if len(images) <= n:
        return images

    # Group by frame folder name (parent dir of the image file)
    from collections import defaultdict

    frames_map = defaultdict(list)
    frame_order = []
    for img_path in images:
        parts = img_path.split("/")
        frame_id = parts[-2] if len(parts) >= 2 else "0"
        if frame_id not in frames_map:
            frame_order.append(frame_id)
        frames_map[frame_id].append(img_path)

    # Cache label detection per frame folder directory
    frame_label_cache = {}

    def frame_has_label(frame_images):
        if not keep_all_labeled:
            return False
        if not frame_images:
            return False
        frame_dir = os.path.dirname(frame_images[0])
        cached = frame_label_cache.get(frame_dir)
        if cached is not None:
            return cached
        for pattern in label_detection_patterns or []:
            found = glob.glob(os.path.join(frame_dir, pattern))
            if found:
                frame_label_cache[frame_dir] = True
                return True
        frame_label_cache[frame_dir] = False
        return False

    # Identify labeled vs unlabeled frames (by order)
    labeled_frames = []
    unlabeled_frames = []
    for fid in frame_order:
        g = frames_map[fid]
        if frame_has_label(g):
            labeled_frames.append(fid)
        else:
            unlabeled_frames.append(fid)

    # Start with all labeled frames' images
    selected_images = []
    for fid in labeled_frames:
        selected_images.extend(frames_map[fid])

    if len(selected_images) >= n:
        # Too many labeled images; trim to n but keep frame integrity best-effort.
        # We trim from the end of labeled list.
        trimmed = []
        count = 0
        for fid in labeled_frames:
            g = frames_map[fid]
            if count + len(g) <= n:
                trimmed.extend(g)
                count += len(g)
            else:
                # Add up to remaining from this frame (may split frame in worst case)
                remain = n - count
                if remain > 0:
                    trimmed.extend(g[:remain])
                count = n
                break
        return trimmed

    # Need to fill from unlabeled frames uniformly
    remaining_needed = n - len(selected_images)
    if not unlabeled_frames:
        return selected_images  # nothing more to add

    # Evenly distribute across unlabeled frames by sampling frame indices via linspace
    if remaining_needed >= sum(len(frames_map[f]) for f in unlabeled_frames):
        # Need all remaining frames
        for fid in unlabeled_frames:
            selected_images.extend(frames_map[fid])
        # Trim to exactly n if we overshoot
        return selected_images[:n]

    # Compute approximate number of frames to include based on average images per frame
    # For single-camera datasets, this equals remaining_needed.
    # We'll choose frames by index and then trim extras to match exactly.
    import numpy as np

    num_unlabeled_frames = len(unlabeled_frames)
    # Estimate how many frames to include to reach remaining_needed images
    avg_size = np.mean([len(frames_map[f]) for f in unlabeled_frames])
    approx_frames_needed = max(1, int(round(remaining_needed / max(avg_size, 1))))
    approx_frames_needed = min(approx_frames_needed, num_unlabeled_frames)

    # Uniform frame indices
    frame_idxs = np.linspace(
        0, num_unlabeled_frames - 1, approx_frames_needed, dtype=int
    )
    chosen_frames = [unlabeled_frames[i] for i in frame_idxs]

    # Aggregate and trim to exact n
    for fid in chosen_frames:
        selected_images.extend(frames_map[fid])
        if len(selected_images) >= n:
            break
    return selected_images[:n]


def chunk_sampling_grouped(frames, n):
    """Group-aware chunk sampling.

    Args:
        frames (List[List[str]]): Ordered list of frame groups (each group is list of image paths for all selected cameras at that timestamp).
        n (int): Target maximum number of individual images per chunk.

    Returns:
        List[List[str]]: Chunks where each chunk is a flattened list of image paths, preserving frame group integrity.

    Strategy:
        Similar to chunk_sampling but operates over frame groups, never splitting a frame across chunks.
    """
    # Flatten size accounting
    frame_sizes = [len(g) for g in frames]
    total_images = sum(frame_sizes)
    if total_images <= n:
        return [sum(frames, [])]

    # Determine number of chunks (ceil)
    num_chunks = (total_images + n - 1) // n
    base = total_images // num_chunks
    remainder = total_images % num_chunks

    chunks = []
    current_chunk = []
    current_size = 0
    chunk_index = 0
    target_size = base + (1 if chunk_index < remainder else 0)

    for frame_group, fsize in zip(frames, frame_sizes):
        # If adding this frame would exceed target_size and we still have room for more chunks, start new chunk.
        if (
            current_size > 0
            and current_size + fsize > target_size
            and len(chunks) + 1 < num_chunks
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
            chunk_index += 1
            target_size = base + (1 if chunk_index < remainder else 0)
        current_chunk.extend(frame_group)
        current_size += fsize

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def load_images(config):
    """
    Loads and samples images from a dataset directory according to the specified configuration.
    Args:
        config (dict): Configuration dictionary containing:
            - "dataset" (dict): Must include "dataset_dir" (str), the root directory of the dataset with structure root/split/drive/image_folders/images.
            - "sampling" (dict): Must include:
                - "strategy" (str): Sampling strategy to use. One of {"chunk", "random", "uniform"}.
                - "num_images" (int): Number of images to sample.
    Returns:
        list: List of chunks, where each chunk contains image file paths.
    Raises:
        ValueError: If an unknown sampling strategy is specified in the configuration.
    """
    dataset_dir = config["dataset"]["dataset_dir"]
    strategy = config["sampling"]["strategy"]
    num_images = config["sampling"]["num_images"]
    num_cameras = config["dataset"].get("num_cameras", 1)
    # Allow explicit camera selection regardless of how many total cameras exist.
    # camera_to_use: int, legacy single selection; cameras_to_use: list for multiple.
    camera_to_use = config["dataset"].get("camera_to_use", None)
    cameras_to_use = config["dataset"].get("cameras_to_use", None)
    splits = config["dataset"].get("splits", ["training", "validation"])
    drives = config["dataset"].get("drives", None)
    file_extension = config["dataset"].get("file_extension", "png")

    # Additional sampling modifiers
    n_interval = config["sampling"].get("n_interval", 1)
    keep_all_labeled = config["sampling"].get("keep_all_labeled", False)
    # Chunk-level filtering based on number of masks present in the images' directories
    min_masks_per_chunk = config["sampling"].get("min_masks_per_chunk", 0)
    mask_patterns = config["sampling"].get(
        "mask_patterns",
        ["mask_*", "*_mask.*", "label_*", "semantic.*"],
    )  # Glob patterns searched per image directory
    # Patterns used to decide if a frame is "labeled" when keep_all_labeled is True.
    # If not explicitly provided, we merge default label patterns with mask_patterns.
    label_detection_patterns = config["sampling"].get(
        "label_detection_patterns",
        None,
    )
    if label_detection_patterns is None:
        # Merge while preserving order and uniqueness
        base_label_patterns = ["label_*", "*_label.*", "semantic.*", "*_seg.*"]
        seen = set()
        merged = []
        for patt in base_label_patterns + list(mask_patterns):
            if patt not in seen:
                merged.append(patt)
                seen.add(patt)
        label_detection_patterns = merged

    SAMPLING_STRATEGIES = {
        "chunk": chunk_sampling,
        "random": random_sampling,
        "uniform": uniform_sampling,
    }

    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown sampling strategy: {strategy} "
            + f"(must be one of {list(SAMPLING_STRATEGIES.keys())})"
        )

    # Collect all images from drives and splits
    all_chunks = []

    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory {split_dir} does not exist")
            continue

        # Get all drives in this split (or use specified drives)
        if drives is None:
            available_drives = [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        else:
            available_drives = drives

        # available_drives = ["4575389405178805994_4900_000_4920_000"]

        for drive in available_drives:
            drive_dir = os.path.join(split_dir, drive)
            if not os.path.exists(drive_dir):
                print(f"Warning: Drive directory {drive_dir} does not exist")
                continue

            drive_images = []

            # Determine camera ids to iterate.
            if cameras_to_use is not None:
                # Normalize list
                if isinstance(cameras_to_use, (int, str)):
                    cam_ids = [int(cameras_to_use)]
                else:
                    cam_ids = [int(c) for c in cameras_to_use]
            elif camera_to_use is not None:
                cam_ids = [int(camera_to_use)]
            else:
                cam_ids = list(range(1, num_cameras + 1))

            # Validate camera ids fall within declared range (warn but continue if not)
            for cid in cam_ids:
                if cid < 1 or cid > num_cameras:
                    print(
                        f"Warning: requested camera id {cid} outside 1..{num_cameras}; skipping."
                    )
            cam_ids = [cid for cid in cam_ids if 1 <= cid <= num_cameras]
            if not cam_ids:
                print(
                    f"Warning: no valid camera ids after filtering for drive {drive}; using all cameras."
                )
                cam_ids = list(range(1, num_cameras + 1))

            for cam_id in cam_ids:
                image_pattern = os.path.join(
                    drive_dir, "**", f"image_{cam_id}.{file_extension}"
                )
                drive_images.extend(glob.glob(image_pattern, recursive=True))

            # Sort images using the existing heuristic
            drive_images.sort(key=sort_function)

            # Group images by frame (frame folder index) so interval filtering keeps camera groups together
            if n_interval > 1 and drive_images:
                frames = defaultdict(list)
                frame_order = []
                for img_path in drive_images:
                    parts = img_path.split("/")
                    # Defensive: ensure we have at least two parent folders
                    frame_id = parts[-2] if len(parts) >= 2 else "0"
                    if frame_id not in frames:
                        frame_order.append(frame_id)
                    frames[frame_id].append(img_path)
                # Cache label detection per frame folder to avoid repeated globs
                frame_label_cache = {}

                def frame_has_label(frame_images):
                    """Decide if a frame has label/mask data using configured patterns."""
                    if not keep_all_labeled:
                        return False
                    # All images in a frame share the same parent directory (frame folder)
                    if not frame_images:
                        return False
                    frame_dir = os.path.dirname(frame_images[0])
                    cached = frame_label_cache.get(frame_dir)
                    if cached is not None:
                        return cached
                    for pattern in label_detection_patterns:
                        found = glob.glob(os.path.join(frame_dir, pattern))
                        if found:
                            frame_label_cache[frame_dir] = True
                            return True
                    frame_label_cache[frame_dir] = False
                    return False

                kept_drive_images = []
                rescued_frames = 0
                for idx, frame_id in enumerate(frame_order):
                    frame_images = frames[frame_id]
                    use_frame = idx % n_interval == 0
                    if (
                        not use_frame
                        and keep_all_labeled
                        and frame_has_label(frame_images)
                    ):
                        use_frame = True
                        rescued_frames += 1
                    if use_frame:
                        kept_drive_images.extend(frame_images)
                if keep_all_labeled and rescued_frames > 0:
                    print(
                        f"keep_all_labeled: rescued {rescued_frames} additional frame(s) from interval skipping in drive {drive}"
                    )
                drive_images = kept_drive_images

            if drive_images:
                # Apply sampling strategy to this drive's images.
                if strategy == "chunk" and len(cam_ids) > 1:
                    # Build frame groups (already computed earlier if interval applied; recompute if needed)
                    frames_map = defaultdict(list)
                    frame_order = []
                    for img_path in drive_images:
                        parts = img_path.split("/")
                        frame_id = parts[-2] if len(parts) >= 2 else "0"
                        if frame_id not in frames_map:
                            frame_order.append(frame_id)
                        frames_map[frame_id].append(img_path)
                    ordered_frames = [frames_map[fid] for fid in frame_order]
                    drive_chunks = chunk_sampling_grouped(ordered_frames, num_images)
                elif strategy == "chunk":
                    # Single-camera chunking: return one chunk per up-to-n images (n=1 -> per-image chunks)
                    drive_chunks = chunk_sampling(drive_images, num_images)
                elif strategy == "uniform":
                    # Use group-aware sampling so we never split frame folders; single chunk per drive
                    selected = uniform_sampling_grouped_keep_labels(
                        drive_images,
                        num_images,
                        keep_all_labeled,
                        label_detection_patterns,
                    )
                    drive_chunks = [selected]
                else:
                    # Random or other non-chunk strategies: single chunk per drive
                    sampled = SAMPLING_STRATEGIES[strategy](drive_images, num_images)
                    drive_chunks = [sampled]
                # Optional filtering of chunks by mask availability
                if min_masks_per_chunk > 0:
                    filtered_chunks = []
                    dropped = 0
                    for ch in drive_chunks:
                        mask_files = set()
                        for img in ch:
                            img_dir = os.path.dirname(img)
                            for patt in mask_patterns:
                                for mf in glob.glob(os.path.join(img_dir, patt)):
                                    mask_files.add(mf)
                        if len(mask_files) >= min_masks_per_chunk:
                            filtered_chunks.append(ch)
                        else:
                            dropped += 1
                    if dropped:
                        print(
                            f"Filtering: dropped {dropped} / {len(drive_chunks)} chunks (< {min_masks_per_chunk} masks) for drive {drive}"
                        )
                    drive_chunks = filtered_chunks

                all_chunks.extend(drive_chunks)
                print(
                    f"Drive {drive} in {split}: {len(drive_images)} images -> {len(drive_chunks)} chunks"
                )
            else:
                print(f"Warning: No images found in drive {drive} of split {split}")

    return all_chunks
