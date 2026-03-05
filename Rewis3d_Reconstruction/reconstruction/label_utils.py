"""
Label mapping utilities for KITTI360 dataset.
Creates various ID mappings and color maps from the config file.
"""


def create_label_mappings(config):
    """
    Create label mappings from config.

    Args:
        config (dict): Configuration dictionary with labels section

    Returns:
        dict: Dictionary containing all label mappings
    """
    ignore_index = config["labels"]["ignore_index"]
    labels = config["labels"]["class_definitions"]

    # Convert to tuple format: (name, id, train_id, color)
    label_tuples = [(label[0], label[1], label[2], tuple(label[3])) for label in labels]

    # Basic mappings
    id2label_2d = {label[1]: label for label in label_tuples}
    trainId2label_2d = {label[2]: label for label in reversed(label_tuples)}
    id2trainId_2d = {label[1]: label[2] for label in label_tuples}
    trainId2color_2d = {label[2]: label[3] for label in label_tuples}
    class_labels_2d = {label[2]: label[0] for label in reversed(label_tuples)}
    colormap_2d = {label[2]: label[3] for label in label_tuples}

    # 3D mappings (sky gets ignored)
    id2trainId_3d = id2trainId_2d.copy()
    id2trainId_3d[23] = ignore_index  # Sky -> ignore_index

    class_labels_3d = class_labels_2d.copy()
    if 15 in class_labels_3d:  # Remove sky from 3D labels
        del class_labels_3d[15]

    colormap_3d = colormap_2d
    id2trainId_original_segment = id2trainId_2d

    return {
        "ignore_index": ignore_index,
        "labels": label_tuples,
        "id2label_2d": id2label_2d,
        "trainId2label_2d": trainId2label_2d,
        "id2trainId_2d": id2trainId_2d,
        "trainId2color_2d": trainId2color_2d,
        "class_labels_2d": class_labels_2d,
        "colormap_2d": colormap_2d,
        "id2trainId_3d": id2trainId_3d,
        "class_labels_3d": class_labels_3d,
        "colormap_3d": colormap_3d,
        "id2trainId_original_segment": id2trainId_original_segment,
    }


def get_class_colors(config):
    """Get class colors for visualization."""
    mappings = create_label_mappings(config)
    return mappings["trainId2color_2d"]


def convert_ids_to_train_ids(label_ids, config, mode="2d"):
    """
    Convert label IDs to training IDs.

    Args:
        label_ids: Array of label IDs
        config: Configuration dictionary
        mode: '2d' or '3d' mapping mode

    Returns:
        Array of training IDs
    """
    mappings = create_label_mappings(config)

    if mode == "2d":
        id_to_train_id = mappings["id2trainId_2d"]
    elif mode == "3d":
        id_to_train_id = mappings["id2trainId_3d"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert using vectorized lookup
    import numpy as np

    train_ids = np.array(
        [
            id_to_train_id.get(id_val, mappings["ignore_index"])
            for id_val in label_ids.flatten()
        ]
    )

    return train_ids.reshape(label_ids.shape)
