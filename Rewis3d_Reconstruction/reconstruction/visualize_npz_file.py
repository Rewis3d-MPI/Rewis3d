import numpy as np
import open3d as o3d
import yaml
from label_utils import create_label_mappings

# Load config for label mappings
config_path = "config/test.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

path = "/KITTI360MapAnything100Chunk10Conf70ratiov2/training/2013_05_28_drive_0009_sync_9046_cam1_image_1.npz"

data = np.load(path, allow_pickle=True)

# Extract 3D data
data_3d = data["data_3d"].item()  # Convert from numpy array to dict
data_2d = data["data_2d"].item()  # Convert from numpy array to dict

print("Available 3D keys:", list(data_3d.keys()))
print("Available 2D keys:", list(data_2d.keys()))

# Create label mappings
label_mappings = create_label_mappings(config)

# Extract pointcloud data
points = data_3d["student_coord"]  # (N, 3)
colors = data_3d["student_colors"]  # Normalize to [0, 1] for Open3D
confidence = data_3d["conf"]  # (N,)
student_segments = data_3d["student_segment"]  # (N,)
original_segments = data_3d["original_segment"]  # (N,)


def map_ids(array, id2trainId):
    """Map segment IDs to training IDs using the provided mapping."""
    mask_array = array
    mapped_segmentation_map = np.copy(mask_array)
    for original_value, new_value in id2trainId.items():
        mapped_segmentation_map[mask_array == original_value] = new_value
    return mapped_segmentation_map


print(f"Pointcloud shape: {points.shape}")
print(f"Colors shape: {colors.shape}")
print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
print(f"Student segments unique values: {np.unique(student_segments)}")
print(f"Original segments unique values: {np.unique(original_segments)}")

# Debug: Check if we have meaningful segment labels
non_zero_student = np.sum(student_segments != 0)
non_zero_original = np.sum(original_segments != 0)
print(
    f"Non-zero student segments: {non_zero_student} / {len(student_segments)} ({100 * non_zero_student / len(student_segments):.1f}%)"
)
print(
    f"Non-zero original segments: {non_zero_original} / {len(original_segments)} ({100 * non_zero_original / len(original_segments):.1f}%)"
)

# Check 2D correspondences
point_indices_array = data_2d["point_indices_array"]
pixel_coords_array = data_2d["pixel_coords_array"]
valid_correspondences = point_indices_array >= 0
print(
    f"Valid correspondences: {np.sum(valid_correspondences)} / {len(point_indices_array)} ({100 * np.sum(valid_correspondences) / len(point_indices_array):.1f}%)"
)

# Create Open3D pointcloud
pcd = o3d.geometry.PointCloud()


# Subsample to at most 120000 points (update all related arrays and correspondences)
max_points = 40000
N = len(points)

if N > max_points:
    rng = np.random.default_rng(0)  # deterministic subsample; change seed if desired
    keep_idx = rng.choice(N, size=max_points, replace=False)
    keep_idx.sort()
    keep_idx = point_indices_array

    points = points[keep_idx]
    colors = colors[keep_idx]
    confidence = confidence[keep_idx]
    student_segments = student_segments[keep_idx]
    original_segments = original_segments[keep_idx]
    # Update 2D correspondences to account for removed 3D points
    # Build a mapping old_index -> new_index (or -1 if removed)
    new_idx = np.full(N, -1, dtype=np.int64)
    new_idx[keep_idx] = np.arange(len(keep_idx), dtype=np.int64)

    # Remap point_indices_array entries that reference removed points to -1
    pi = point_indices_array
    try:
        orig_shape = pi.shape
        flat = np.asarray(pi).reshape(-1).copy()

        # Prepare an int array defaulting to -1
        flat_mapped = np.full(flat.shape, -1, dtype=np.int64)

        # Mask of entries that look like valid indices (>=0 and finite)
        with np.errstate(invalid="ignore"):
            valid_mask = (flat >= 0) & np.isfinite(flat)

        if valid_mask.any():
            flat_vals = flat[valid_mask].astype(np.int64)
            # Clip out-of-range to -1
            in_range = (flat_vals >= 0) & (flat_vals < N)
            flat_vals[~in_range] = -1
            # Map using new_idx (values of -1 will produce -1)
            mapped_vals = np.where(flat_vals >= 0, new_idx[flat_vals], -1)
            flat_mapped[valid_mask] = mapped_vals

        # Restore shape
        point_indices_array = flat_mapped.reshape(orig_shape)
    except Exception:
        # On any error, invalidate all correspondences
        point_indices_array = np.full_like(pi, -1)

    # Recompute valid_correspondences based on updated point indices
    valid_correspondences = point_indices_array >= 0


pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize the pointcloud
print("Visualizing pointcloud with original colors...")
o3d.visualization.draw_geometries(
    [pcd], window_name="Student Pointcloud with Colors", width=1024, height=768
)


# Optional: Visualize with confidence-based coloring
def visualize_with_confidence():
    pcd_conf = o3d.geometry.PointCloud()
    pcd_conf.points = o3d.utility.Vector3dVector(points)

    # Create confidence-based colors (blue = low confidence, red = high confidence)
    conf_normalized = (confidence - confidence.min()) / (
        confidence.max() - confidence.min()
    )
    conf_colors = np.zeros((len(points), 3))
    conf_colors[:, 0] = conf_normalized  # Red channel
    conf_colors[:, 2] = 1 - conf_normalized  # Blue channel

    pcd_conf.colors = o3d.utility.Vector3dVector(conf_colors)

    print("Visualizing pointcloud with confidence coloring (red=high, blue=low)...")
    o3d.visualization.draw_geometries(
        [pcd_conf],
        window_name="Pointcloud with Confidence Coloring",
        width=1024,
        height=768,
    )


# Optional: Visualize with segment-based coloring using correct label colors
def visualize_with_segments():
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(points)

    # Map student segments from ID to trainID
    mapped_student_segments = map_ids(student_segments, label_mappings["id2trainId_2d"])

    # Get color mapping
    trainId2color = label_mappings["trainId2color_2d"]
    ignore_index = label_mappings["ignore_index"]

    # Create segment-based colors using the correct color mapping
    seg_colors = np.zeros((len(points), 3))

    for point_idx, train_id in enumerate(mapped_student_segments):
        if train_id in trainId2color and train_id != ignore_index:
            # Use the defined color for this class, normalized to [0, 1]
            seg_colors[point_idx] = np.array(trainId2color[train_id]) / 255.0
        else:
            # Use black for ignore_index or undefined classes
            seg_colors[point_idx] = [0, 0, 0]

    pcd_seg.colors = o3d.utility.Vector3dVector(seg_colors)

    print("Visualizing pointcloud with semantic segment coloring (KITTI360 colors)...")
    o3d.visualization.draw_geometries(
        [pcd_seg],
        window_name="Pointcloud with Semantic Segment Coloring",
        width=1024,
        height=768,
    )


# Optional: Visualize original segments with correct colors
def visualize_with_original_segments():
    pcd_seg = o3d.geometry.PointCloud()
    pcd_seg.points = o3d.utility.Vector3dVector(points)

    # Map original segments from ID to trainID
    mapped_original_segments = map_ids(
        original_segments, label_mappings["id2trainId_original_segment"]
    )

    # Get color mapping
    trainId2color = label_mappings["trainId2color_2d"]
    ignore_index = label_mappings["ignore_index"]

    # Create segment-based colors using the correct color mapping
    seg_colors = np.zeros((len(points), 3))

    for point_idx, train_id in enumerate(mapped_original_segments):
        if train_id in trainId2color and train_id != ignore_index:
            # Use the defined color for this class, normalized to [0, 1]
            seg_colors[point_idx] = np.array(trainId2color[train_id]) / 255.0
        else:
            # Use black for ignore_index or undefined classes
            seg_colors[point_idx] = [0, 0, 0]

    pcd_seg.colors = o3d.utility.Vector3dVector(seg_colors)

    print("Visualizing pointcloud with original segment coloring (KITTI360 colors)...")
    o3d.visualization.draw_geometries(
        [pcd_seg],
        window_name="Pointcloud with Original Segment Coloring",
        width=1024,
        height=768,
    )


# Optional: Visualize valid correspondences highlighted
def visualize_with_valid_correspondences(
    base_gray=0.1, highlight_color=(0.0, 1.0, 0.0)
):
    pcd_corr = o3d.geometry.PointCloud()
    pcd_corr.points = o3d.utility.Vector3dVector(points)

    # Prepare base colors (dark gray to black)
    colors_corr = np.full((len(points), 3), base_gray, dtype=np.float64)

    # Flatten arrays to be robust to shape (1D/2D)
    if point_indices_array.ndim > 1:
        flat_indices = point_indices_array.reshape(-1)
        flat_valid = valid_correspondences.reshape(-1)
    else:
        flat_indices = point_indices_array
        flat_valid = valid_correspondences

    # Extract and sanitize valid point indices
    indices = flat_indices[flat_valid].astype(np.int64)
    indices = indices[(indices >= 0) & (indices < len(points))]
    unique_indices = np.unique(indices)

    if unique_indices.size == 0:
        print("No valid correspondences to highlight.")
    else:
        colors_corr[unique_indices] = np.array(highlight_color, dtype=np.float64)

    pcd_corr.colors = o3d.utility.Vector3dVector(colors_corr)

    print(
        f"Visualizing {unique_indices.size} points with valid correspondences highlighted..."
    )
    o3d.visualization.draw_geometries(
        [pcd_corr],
        window_name="Valid Correspondences Highlighted",
        width=1024,
        height=768,
    )


# New: Visualize correspondences projected onto the image
def visualize_correspondences_on_image(
    highlight_color=(0, 0, 255),  # BGR (red)
    radius=2,
    thickness=-1,
    show_window=True,
    save_path=None,
):
    # Try to load image either from data_2d array or from a path
    img = None
    if "student_image_1" in data_2d:
        img = data_2d["student_image_1"]
        if isinstance(img, np.ndarray):
            arr = img
            # Normalize/convert to uint8
            if arr.dtype != np.uint8:
                arr_min, arr_max = float(arr.min()), float(arr.max())
                if arr_max > 1.0 or arr_min < 0.0:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                else:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            # Ensure 3-channel BGR for OpenCV
            if arr.ndim == 2:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # Assume RGB, convert to BGR
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                print("Image in data_2d has unexpected shape; cannot visualize.")
                return
            img = img_bgr
        else:
            img = None
    elif "image_path" in data_2d or "img_path" in data_2d or "rgb_path" in data_2d:
        p = data_2d.get("image_path", data_2d.get("img_path", data_2d.get("rgb_path")))
        if isinstance(p, np.ndarray):
            try:
                p = p.item()
            except Exception:
                pass
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load image at path: {p}")
            return
    else:
        print(
            "No image or image path found in data_2d; cannot visualize correspondences on image."
        )
        return

    # Convert base to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        print("Unexpected image shape; cannot visualize.")
        return

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = gray.shape[:2]

    # Select only valid correspondences
    coords = pixel_coords_array[valid_correspondences]
    coords = np.asarray(coords)

    if coords.ndim != 2 or coords.shape[1] < 2:
        print("pixel_coords_array has unexpected shape; skipping drawing.")
        return

    # Treat coords as (y, x) => row, col
    xs = np.rint(coords[:, 1]).astype(np.int32)  # x from second component (width/col)
    ys = np.rint(coords[:, 0]).astype(np.int32)  # y from first component (height/row)
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs_inb, ys_inb = xs[inb], ys[inb]

    if xs_inb.size == 0:
        print("No in-bounds pixel coordinates to highlight.")
    else:
        pts = np.stack([xs_inb, ys_inb], axis=1)
        # Deduplicate
        pts = np.unique(pts, axis=0)
        for x, y in pts:
            cv2.circle(vis, (int(x), int(y)), radius, highlight_color, thickness)

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"Saved correspondence visualization to: {save_path}")

    if show_window:
        cv2.imshow("Image with Valid Correspondences", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Uncomment to see additional visualizations
# visualize_with_confidence()
visualize_with_segments()
visualize_with_original_segments()
visualize_with_valid_correspondences()
visualize_correspondences_on_image(highlight_color=(0, 255, 0), radius=1, thickness=-1)
