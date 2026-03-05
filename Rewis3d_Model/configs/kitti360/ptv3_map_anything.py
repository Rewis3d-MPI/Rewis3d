# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

_base_ = ["../../_base_/default_runtime.py"]

# Weight loading (use -w flag in train.sh instead)
weight = None
# Resume training (use -r flag in train.sh instead)
resume = False

dataset_samples = 41495

# misc custom setting
batch_size = 14  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
num_worker = 4

use_cmc_loss = True
cmc_epoch = 15

log_train_image_data = False

cmc_ramp_epochs = 5
cmc_max_weight_2d = 0.075
cmc_max_weight_3d = 0.125

save_epochs = [15]

steps_per_epoch = dataset_samples // batch_size

image_slices = [
    (0, 512),
    (192, 704),
    (448, 960),
    (704, 1216),
    (896, 1408),
]
logit_slices = [(0, 128), (48, 176), (112, 240), (176, 304), (224, 352)]

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
    criteria=dict(type="CMCSkyTeacherLoss"),
)

# model settings
model_2d = dict(
    type="Segmentation2DModel",
    domain="2D",
    model_class="student_teacher",
    num_classes=16,
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
    num_classes=15,
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
epoch = 35
eval_epoch = 35

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
data_root = "data/KITTI360_MA"
ignore_index = 255

# Name ID TrainID Color
labels = [
    ("unlabeled", 0, ignore_index, (0, 0, 0)),
    ("ego vehicle", 1, ignore_index, (0, 0, 0)),
    ("rectification border", 2, ignore_index, (0, 0, 0)),
    ("out of roi", 3, ignore_index, (0, 0, 0)),
    ("static", 4, ignore_index, (0, 0, 0)),
    ("dynamic", 5, ignore_index, (111, 74, 0)),
    ("ground", 6, ignore_index, (81, 0, 81)),
    ("road", 7, 0, (128, 64, 128)),
    ("sidewalk", 8, 1, (244, 35, 232)),
    ("parking", 9, ignore_index, (250, 170, 160)),
    ("rail track", 10, ignore_index, (230, 150, 140)),
    ("building", 11, 2, (70, 70, 70)),
    ("wall", 12, 3, (102, 102, 156)),
    ("fence", 13, 4, (190, 153, 153)),
    ("guard rail", 14, ignore_index, (180, 165, 180)),
    ("bridge", 15, ignore_index, (150, 100, 100)),
    ("tunnel", 16, ignore_index, (150, 120, 90)),
    ("pole", 17, 5, (153, 153, 153)),
    ("polegroup", 18, ignore_index, (153, 153, 153)),
    ("traffic light", 19, 6, (250, 170, 30)),
    ("traffic sign", 20, 7, (220, 220, 0)),
    ("vegetation", 21, 8, (107, 142, 35)),
    ("terrain", 22, 9, (152, 251, 152)),
    ("sky", 23, 15, (70, 130, 180)),
    ("person", 24, 10, (220, 20, 60)),
    ("rider", 25, ignore_index, (255, 0, 0)),
    ("car", 26, 11, (0, 0, 142)),
    ("truck", 27, 12, (0, 0, 70)),
    ("bus", 28, ignore_index, (0, 60, 100)),
    ("caravan", 29, ignore_index, (0, 0, 90)),
    ("trailer", 30, ignore_index, (0, 0, 110)),
    ("train", 31, ignore_index, (0, 80, 100)),
    ("motorcycle", 32, 13, (0, 0, 230)),
    ("bicycle", 33, 14, (119, 11, 32)),
    ("garage", 34, 2, (64, 128, 128)),
    ("gate", 35, 4, (190, 153, 153)),
    ("stop", 36, ignore_index, (150, 120, 90)),
    ("smallpole", 37, 5, (153, 153, 153)),
    ("lamp", 38, ignore_index, (0, 64, 64)),
    ("trash bin", 39, ignore_index, (0, 128, 192)),
    ("vending machine", 40, ignore_index, (128, 64, 0)),
    ("box", 41, ignore_index, (64, 64, 128)),
    ("unknown construction", 42, ignore_index, (102, 0, 0)),
    ("unknown vehicle", 43, ignore_index, (51, 0, 51)),
    ("unknown object", 44, ignore_index, (32, 32, 32)),
    ("license plate", 45, ignore_index, (0, 0, 142)),
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
    num_classes=16,
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
            dict(type="RandCrop", crop_size=(376, 512)),
            dict(type="Blur", student_only=True),
            dict(type="AugMix", student_only=True),
            dict(type="Cutout", p=1, student_only=True),
            dict(type="Cutout", p=0.5, student_only=True),
            dict(type="ExtractFeatures", height=376, width=512),
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
            dict(type="ExtractFeatures", height=376, width=1408),
            dict(type="ToTensor"),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
)
