import argparse
import json
import os
import time

import yaml
from tqdm import tqdm

from .create_reconstructions import generate_reconstruction
from .image_loading import load_images
from .pointcloud_processing import create_unified_pointcloud
from .save_dataset import generate_output_filename, save_dataset_for_chunk


def parse_config():
    parser = argparse.ArgumentParser(description="Dataset generation config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2'). If not specified, uses 0 to num_gpus-1",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    return config, args


def main():
    config, args = parse_config()

    num_gpus = args.num_gpus

    # Derive a camera signature from config to make checkpoints camera-aware
    def _get_camera_signature(cfg: dict) -> str:
        ds = cfg.get("dataset", {})
        cams = ds.get("cameras_to_use")
        # Normalize to a sorted list of ints if possible
        if cams is None or cams == []:
            return "cams-all"  # treat as all cameras (or unspecified)
        if isinstance(cams, int):
            cams_list = [cams]
        else:
            # Try to coerce any elements to int where possible
            try:
                cams_list = [int(c) for c in cams]
            except Exception:
                # Fallback to string signature
                return f"cams-{str(cams)}"
        cams_list = sorted(cams_list)
        return "cams-" + "-".join(str(c) for c in cams_list)

    camera_sig = _get_camera_signature(config)

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(num_gpus))

    if len(gpu_ids) != num_gpus:
        raise ValueError(
            f"Number of GPU IDs ({len(gpu_ids)}) doesn't match num_gpus ({num_gpus})"
        )

    # Step 1: Load and sample images
    tqdm.write("Loading and sampling images…")
    images_chunks = load_images(config)
    num_chunks = len(images_chunks)
    tqdm.write(f"Prepared {num_chunks} chunks")
    tqdm.write(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # Checkpoint directory inside output_dir
    output_dir = config["dataset"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, "_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def expected_npz_paths(chunk_images, labeled_mask=None):
        # Absolute paths in output_dir for each expected NPZ of the chunk.
        # If labeled_mask (list[bool]) is provided, only include images with True.
        selected = (
            [p for p, keep in zip(chunk_images, labeled_mask) if keep]
            if labeled_mask is not None
            else chunk_images
        )
        return [
            os.path.join(output_dir, generate_output_filename(p, config))
            for p in selected
        ]

    def _extract_split(image_path: str):
        parts = image_path.split(os.sep)
        # Expect .../<dataset_dir>/<split>/<drive>/<image_folder>/<image_file>
        # So split should be -4 relative to file path end
        return parts[-4] if len(parts) >= 4 else "unknown"

    def _split_tagged_fname(split, chunk_idx, suffix):
        # Include camera signature in filenames so checkpoints are camera-aware
        return f"{split}_{camera_sig}_chunk_{chunk_idx:05d}.{suffix}"

    def is_chunk_completed(chunk_idx, chunk_images):
        split = _extract_split(chunk_images[0]) if chunk_images else "unknown"
        done_flag = os.path.join(
            ckpt_dir, _split_tagged_fname(split, chunk_idx, "done")
        )
        if os.path.exists(done_flag):
            return True
        # Backward compatibility note:
        # Previously, checkpoints did not encode camera selection and used either
        #   - split-less legacy: "chunk_00000.done"
        #   - split-tagged without camera signature: "{split}_chunk_00000.done"
        # To ensure reruns with different camera selections are not skipped,
        # we DO NOT treat legacy markers as completed for the current camera signature.
        # However, we still record their presence into the JSON when marking completed.
        return False

    def mark_chunk_completed(chunk_idx, chunk_images, extra=None):
        split = _extract_split(chunk_images[0]) if chunk_images else "unknown"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = {
            "chunk_idx": chunk_idx,
            "split": split,
            "num_images": len(chunk_images),
            "timestamp": ts,
            "images": chunk_images,
        }
        if extra:
            meta.update(extra)
        json_path = os.path.join(
            ckpt_dir, _split_tagged_fname(split, chunk_idx, "json")
        )
        with open(json_path, "w") as f:
            # Include camera signature for traceability
            meta["camera_signature"] = camera_sig
            json.dump(meta, f, indent=2)
        # Touch .done marker (split-tagged)
        open(
            os.path.join(ckpt_dir, _split_tagged_fname(split, chunk_idx, "done")), "w"
        ).close()

    if num_gpus == 1:
        # Single GPU mode - original sequential processing
        process_chunks_single_gpu(
            images_chunks, config, is_chunk_completed, mark_chunk_completed, gpu_ids[0]
        )
    else:
        # Multi-GPU mode - parallel processing
        process_chunks_multi_gpu(
            images_chunks,
            config,
            is_chunk_completed,
            mark_chunk_completed,
            gpu_ids,
            num_gpus,
            camera_sig,
        )


def process_chunks_single_gpu(
    images_chunks, config, is_chunk_completed, mark_chunk_completed, device_id
):
    """Process chunks sequentially on a single GPU."""
    num_chunks = len(images_chunks)
    reconstructions = []

    for chunk_idx, chunk_images in enumerate(
        tqdm(images_chunks, desc="Chunks", unit="chunk")
    ):
        # Skip if already completed
        if is_chunk_completed(chunk_idx, chunk_images):
            tqdm.write(
                f"Skipping chunk {chunk_idx + 1}/{num_chunks}: already completed"
            )
            continue

        try:
            # Step 2: Generate reconstructions for this chunk
            predictions = generate_reconstruction(
                config, chunk_images, device_id=device_id
            )

            # Step 3: Create unified pointcloud and unproject labels
            unified_data = create_unified_pointcloud(predictions, chunk_images, config)

            # Step 4: Apply view-aware sampling and save dataset (only labeled views are saved inside)
            save_dataset_for_chunk(unified_data, predictions, config)

            # Record success checkpoint
            mark_chunk_completed(
                chunk_idx,
                chunk_images,
                extra={"num_views": len(chunk_images), "gpu_id": device_id},
            )

            # Keep minimal in-memory record
            reconstructions.append(
                {
                    "chunk_idx": chunk_idx,
                    "image_paths": chunk_images,
                    "num_views": len(chunk_images),
                }
            )

        except Exception as e:
            tqdm.write(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
            import traceback

            tqdm.write(traceback.format_exc())
            continue

    tqdm.write(
        f"Finished. Successful chunks this run: {len(reconstructions)} / {num_chunks}"
    )


def worker_process_chunks(
    gpu_id, chunk_assignments, images_chunks, config, ckpt_dir, output_dir, camera_sig
):
    """
    Worker function for processing chunks on a specific GPU.

    Args:
        gpu_id (int): GPU device ID for this worker
        chunk_assignments (list): List of chunk indices assigned to this worker
        images_chunks (list): All image chunks
        config (dict): Configuration dictionary
        ckpt_dir (str): Checkpoint directory path
        output_dir (str): Output directory path
    """
    import json
    import os
    import time

    def _extract_split(image_path: str):
        parts = image_path.split(os.sep)
        return parts[-4] if len(parts) >= 4 else "unknown"

    def _split_tagged_fname(split, chunk_idx, suffix):
        return f"{split}_{camera_sig}_chunk_{chunk_idx:05d}.{suffix}"

    def is_chunk_completed(chunk_idx, chunk_images):
        split = _extract_split(chunk_images[0]) if chunk_images else "unknown"
        done_flag = os.path.join(
            ckpt_dir, _split_tagged_fname(split, chunk_idx, "done")
        )
        if os.path.exists(done_flag):
            return True
        return False

    def mark_chunk_completed(chunk_idx, chunk_images, extra=None):
        split = _extract_split(chunk_images[0]) if chunk_images else "unknown"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = {
            "chunk_idx": chunk_idx,
            "split": split,
            "num_images": len(chunk_images),
            "timestamp": ts,
            "images": chunk_images,
        }
        if extra:
            meta.update(extra)
        json_path = os.path.join(
            ckpt_dir, _split_tagged_fname(split, chunk_idx, "json")
        )
        with open(json_path, "w") as f:
            meta["camera_signature"] = camera_sig
            json.dump(meta, f, indent=2)
        open(
            os.path.join(ckpt_dir, _split_tagged_fname(split, chunk_idx, "done")), "w"
        ).close()

    print(f"[GPU {gpu_id}] Worker started, processing {len(chunk_assignments)} chunks")

    successful = 0
    for chunk_idx in chunk_assignments:
        chunk_images = images_chunks[chunk_idx]

        # Skip if already completed
        if is_chunk_completed(chunk_idx, chunk_images):
            print(f"[GPU {gpu_id}] Skipping chunk {chunk_idx}: already completed")
            continue

        try:
            print(
                f"[GPU {gpu_id}] Processing chunk {chunk_idx} ({len(chunk_images)} images)"
            )

            # Step 2: Generate reconstructions for this chunk on assigned GPU
            predictions = generate_reconstruction(
                config, chunk_images, device_id=gpu_id
            )

            # Step 3: Create unified pointcloud and unproject labels
            unified_data = create_unified_pointcloud(predictions, chunk_images, config)

            # Step 4: Apply view-aware sampling and save dataset
            save_dataset_for_chunk(unified_data, predictions, config)

            # Record success checkpoint
            mark_chunk_completed(
                chunk_idx,
                chunk_images,
                extra={"num_views": len(chunk_images), "gpu_id": gpu_id},
            )

            successful += 1
            print(f"[GPU {gpu_id}] Completed chunk {chunk_idx}")

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing chunk {chunk_idx}: {str(e)}")
            import traceback

            print(traceback.format_exc())
            continue

    print(
        f"[GPU {gpu_id}] Worker finished. Processed {successful}/{len(chunk_assignments)} chunks successfully"
    )
    return successful


def process_chunks_multi_gpu(
    images_chunks,
    config,
    is_chunk_completed,
    mark_chunk_completed,
    gpu_ids,
    num_gpus,
    camera_sig,
):
    """Process chunks in parallel across multiple GPUs."""
    import multiprocessing as mp

    num_chunks = len(images_chunks)
    output_dir = config["dataset"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, "_checkpoints")

    # Assign chunks to GPUs in round-robin fashion
    gpu_assignments = [[] for _ in range(num_gpus)]
    for chunk_idx in range(num_chunks):
        gpu_idx = chunk_idx % num_gpus
        gpu_assignments[gpu_idx].append(chunk_idx)

    print("\nChunk distribution across GPUs:")
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        print(f"  GPU {gpu_id}: {len(gpu_assignments[gpu_idx])} chunks")
    print()

    # Use spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    # Create worker processes
    processes = []
    for gpu_idx, gpu_id in enumerate(gpu_ids):
        if len(gpu_assignments[gpu_idx]) == 0:
            continue

        p = mp.Process(
            target=worker_process_chunks,
            args=(
                gpu_id,
                gpu_assignments[gpu_idx],
                images_chunks,
                config,
                ckpt_dir,
                output_dir,
                camera_sig,
            ),
        )
        p.start()
        processes.append(p)
        print(f"Started worker for GPU {gpu_id}")

    # Wait for all workers to complete
    print(f"\nWaiting for {len(processes)} workers to complete...")
    for p in processes:
        p.join()

    print("\nAll workers finished!")

    # Count total completed chunks
    completed_count = 0
    for chunk_idx, chunk_images in enumerate(images_chunks):
        if is_chunk_completed(chunk_idx, chunk_images):
            completed_count += 1

    print(f"Total completed chunks: {completed_count}/{num_chunks}")


if __name__ == "__main__":
    main()
