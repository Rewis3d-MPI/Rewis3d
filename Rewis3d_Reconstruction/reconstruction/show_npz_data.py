import numpy as np
import open3d as o3d
import yaml
from .label_utils import create_label_mappings

# Load config for label mappings
config_path = "reconstruction/config/test.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

path = "/scratch/inf0/user/jernst/Datasets/WaymoMapAnything180Chunk10Conf70ratio4interval/training/11343624116265195592_5910_530_5930_530_1521839103685689_cam1_image_1.npz"

data = np.load(path, allow_pickle=True)

# Extract 3D data
data_3d = data["data_3d"].item()  # Convert from numpy array to dict
data_2d = data["data_2d"].item()  # Convert from numpy array to dict

print("Available 3D keys:", list(data_3d.keys()))
print("Available 2D keys:", list(data_2d.keys()))

# -----------------------------------------------------------------------------
# Added: Per-element statistics (name, shape, dtype, min, max) for each entry
# -----------------------------------------------------------------------------


def print_stats(name, value):
    if isinstance(value, np.ndarray):
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            vmin = value.min()
            vmax = value.max()
        else:
            vmin = "n/a"
            vmax = "n/a"
        print(
            f"{name:25s} shape={value.shape} dtype={value.dtype} min={vmin} max={vmax}"
        )
    else:
        print(f"{name:25s} type={type(value).__name__}")


print("\n=== data_3d contents ===")
for k, v in data_3d.items():
    print_stats(k, v)

print("\n=== data_2d contents ===")
for k, v in data_2d.items():
    print_stats(k, v)

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


# Uncomment to see additional visualizations
# visualize_with_confidence()
# visualize_with_segments()
# visualize_with_original_segments()
