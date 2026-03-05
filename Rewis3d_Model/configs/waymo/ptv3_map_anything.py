# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

_base_ = ["../../_base_/default_runtime.py"]

# Weight loading (use -w flag in train.sh instead)
weight = None
# Resume training (use -r flag in train.sh instead)
resume = False

dataset_samples = 39618

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
num_worker = 4

use_cmc_loss = True
cmc_epoch = 100

log_train_image_data = False

cmc_ramp_epochs = 15
cmc_max_weight_2d = 0.075
cmc_max_weight_3d = 0.15

save_epochs = [14]

steps_per_epoch = dataset_samples // batch_size

image_slices = [(0, 640), (320, 960)]
logit_slices = [(0, 160), (80, 240)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=200),
    dict(type="InformationWriter"),
    dict(type="SemanticEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="SlurmTimeHook"),
]

main_criteria = dict(
    model_class="main_criteria",
    criteria=dict(type="CMCLoss"),
)

# model settings
model_2d = dict(
    type="Segmentation2DModel",
    domain="2D",
    model_class="student_teacher",
    num_classes=25,
    model="nvidia/mit-b4",
    criteria=[
        dict(
            type="PartialConsistencyLoss",
            h_name="cross_entropy",
            ignore_index=255,
            loss_type="weighted_consistency",
        ),
    ],
)

model_3d = dict(
    type="Segmentation3DModel",
    model_class="student_teacher",
    domain="3D",
    num_classes=24,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(
            type="PartialConsistencyLoss3D",
            h_name="cross_entropy",
            ignore_index=255,
            loss_type="weighted_consistency",
        ),
    ],
)

model_list = [model_2d, model_3d]

# scheduler settings
epoch = 50
eval_epoch = 50

total_steps = steps_per_epoch * epoch

optimizer_3d = dict(type="AdamW", lr=0.002, weight_decay=0.005)
optimizer_2d = dict(type="AdamW", lr=0.00003, weight_decay=1e-08)
scheduler_3d = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
# Set as it works like fixed lr
scheduler_2d = dict(type="PolyLR", total_steps=total_steps, power=1.0)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "DefaultReconstructedDataset"
data_root = "data/Waymo_MA"
ignore_index = 255

# Name ID TrainID Color
labels = [
    ("undefined", 0, ignore_index, (0, 0, 0)),
    ("ego vehicle", 1, ignore_index, (0, 0, 0)),  # TYPE_EGO_VEHICLE (The Waymo vehicle)
    ("car", 2, 0, (0, 0, 142)),  # TYPE_CAR (Small vehicles like sedans, SUVs, etc.)
    ("truck", 3, 1, (0, 0, 70)),  # TYPE_TRUCK (Large vehicles carrying cargo)
    ("bus", 4, 2, (0, 60, 100)),  # TYPE_BUS (Vehicles carrying more than 8 passengers)
    (
        "other large vehicle",
        5,
        3,
        (0, 0, 90),
    ),  # TYPE_OTHER_LARGE_VEHICLE (Large vehicles not truck or bus)
    ("bicycle", 6, 4, (119, 11, 32)),  # TYPE_BICYCLE (Bicycle with no rider)
    ("motorcycle", 7, 5, (0, 0, 230)),  # TYPE_MOTORCYCLE (Motorcycle with no rider)
    (
        "trailer",
        8,
        6,
        (0, 0, 110),
    ),  # TYPE_TRAILER (Trailer attached to another vehicle)
    (
        "pedestrian",
        9,
        7,
        (220, 20, 60),
    ),  # TYPE_PEDESTRIAN (Pedestrians without objects)
    ("cyclist", 10, 8, (255, 0, 0)),  # TYPE_CYCLIST (Bicycle with rider)
    ("motorcyclist", 11, 9, (235, 0, 0)),  # TYPE_MOTORCYCLIST (Motorcycle with rider)
    ("bird", 12, 10, (135, 206, 250)),  # TYPE_BIRD (Birds, including on the ground)
    (
        "ground animal",
        13,
        11,
        (139, 69, 19),
    ),  # TYPE_GROUND_ANIMAL (Animals like dogs, cats, cows)
    (
        "construction cone pole",
        14,
        12,
        (133, 133, 133),
    ),  # TYPE_CONSTRUCTION_CONE_POLE (Cone or short pole related to construction)
    ("pole", 15, 13, (153, 153, 153)),  # TYPE_POLE (Lamp post, traffic sign pole, etc.)
    (
        "pedestrian object",
        16,
        14,
        (90, 34, 10),
    ),  # TYPE_PEDESTRIAN_OBJECT (Objects carried/pushed/dragged by a pedestrian)
    ("sign", 17, 15, (220, 220, 0)),  # TYPE_SIGN (Traffic-related signs)
    (
        "traffic light",
        18,
        16,
        (250, 170, 30),
    ),  # TYPE_TRAFFIC_LIGHT (Traffic light boxes)
    ("building", 19, 17, (70, 70, 70)),  # TYPE_BUILDING (Permanent buildings and walls)
    ("road", 20, 18, (128, 64, 128)),  # TYPE_ROAD (Drivable road surfaces)
    (
        "lane marker",
        21,
        22,
        (200, 200, 200),
    ),  # TYPE_LANE_MARKER (Lane-defining markings)
    (
        "road marker",
        22,
        23,
        (180, 180, 180),
    ),  # TYPE_ROAD_MARKER (All other road markings)
    (
        "sidewalk",
        23,
        19,
        (244, 35, 232),
    ),  # TYPE_SIDEWALK (Paved walkable surface, including curbs)
    (
        "vegetation",
        24,
        20,
        (107, 142, 35),
    ),  # TYPE_VEGETATION (Vegetation including tree trunks, branches, bushes, etc.)
    ("sky", 25, 24, (70, 130, 180)),  # TYPE_SKY (Sky, including clouds)
    (
        "ground",
        26,
        21,
        (152, 251, 152),
    ),  # TYPE_GROUND (Other horizontal surfaces that are drivable or walkable)
    (
        "dynamic object",
        27,
        ignore_index,
        (111, 74, 0),
    ),  # TYPE_DYNAMIC (Non-permanent objects)
    ("static object", 28, ignore_index, (0, 0, 0)),  # TYPE_STATIC (Permanent objects)
]


sky_class_name = "sky"
sky_entry = next(label for label in labels if label[0] == sky_class_name)
sky_label = sky_entry[2]
sky_id = sky_entry[1]

id2label_2d = {label[1]: label for label in labels}
trainId2label_2d = {label[2]: label for label in reversed(labels)}
id2trainId_2d = {label[1]: label[2] for label in labels}
trainId2color_2d = {label[2]: label[3] for label in labels}
class_labels_2d = {label[2]: label[0] for label in reversed(labels)}
colormap_2d = {label[2]: label[3] for label in labels}
id2trainId_3d = id2trainId_2d.copy()
id2trainId_3d[sky_id] = ignore_index
class_labels_3d = class_labels_2d.copy()
if sky_label in class_labels_3d:
    del class_labels_3d[sky_label]
colormap_3d = colormap_2d
id2trainId_original_segment = id2trainId_2d

data = dict(
    num_classes=25,
    ignore_index=ignore_index,
    labels_2d=labels,
    labels_3d=labels,
    train=dict(
        type=dataset_type,
        split="training",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict=dict(student_coord="teacher_coord")),
            dict(
                type="MapIds",
                id2trainId_segment=id2trainId_3d,
                id2trainId_original_segment=id2trainId_3d,
            ),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
                student_only=True,
            ),
            dict(type="PointClip", point_cloud_range=(-70, -30, -130, 70, 30, 130)),
            dict(type="RandomScale", scale=[0.9, 1.1], student_only=True),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5, student_only=True),
            dict(type="RandomJitter", sigma=0.005, clip=0.02, student_only=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "student_coord",
                    "student_segment",
                    "teacher_coord",
                    "original_segment",
                    "student_colors",
                    "conf",
                ),
                student_feat_keys=["student_coord", "student_colors"],
                teacher_feat_keys=["teacher_coord", "student_colors"],
            ),
        ],
        transform_2d=[
            dict(type="ToPIL"),
            dict(type="RescaleAndDistort", student_only=True),
            dict(
                type="Copy",
                keys_dict=dict(
                    student_mask_1="teacher_mask_1", student_image_1="teacher_image_1"
                ),
            ),
            dict(
                type="MapIds",
                id2trainId=id2trainId_2d,
                id2trainId_original_mask=id2trainId_2d,
            ),
            dict(type="RandHorizontalFlip"),
            dict(type="RandCrop", crop_size=(640, 640)),
            dict(type="Blur", student_only=True),
            dict(type="AugMix", student_only=True),
            dict(type="Cutout", p=1, student_only=True),
            dict(type="Cutout", p=0.5, student_only=True),
            dict(type="ExtractFeatures", height=640, width=640),
            dict(type="ToTensor"),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            # dict(type="RandomScale", scale=[40, 40], student_only=True),
            dict(type="Copy", keys_dict=dict(student_coord="teacher_coord")),
            # , student_segment="teacher_segment")),
            dict(
                type="MapIds",
                id2trainId_segment=id2trainId_3d,
                id2trainId_original_segment=id2trainId_3d,
            ),
            dict(type="PointClip", point_cloud_range=(-70, -30, -130, 70, 30, 130)),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "student_coord",
                    "student_segment",
                    "teacher_coord",
                    "original_segment",
                    "student_colors",
                ),
                student_feat_keys=["student_coord", "student_colors"],
                teacher_feat_keys=["teacher_coord", "student_colors"],
            ),
        ],
        transform_2d=[
            dict(type="ToPIL"),
            dict(
                type="Copy",
                keys_dict=dict(
                    student_mask_1="teacher_mask_1", student_image_1="teacher_image_1"
                ),
            ),
            dict(
                type="MapIds",
                id2trainId=id2trainId_2d,
                id2trainId_original_mask=id2trainId_2d,
            ),
            dict(type="ExtractFeatures", height=640, width=960),
            dict(type="ToTensor"),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
)
