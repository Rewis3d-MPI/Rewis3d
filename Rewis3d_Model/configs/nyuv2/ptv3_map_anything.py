# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.

_base_ = ["../../_base_/default_runtime.py"]

# Weight loading (use -w flag in train.sh instead)
weight = None
# Resume training (use -r flag in train.sh instead)
resume = False

dataset_samples = 647

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = False
enable_amp = True
num_worker = 4

use_cmc_loss = True
cmc_epoch = 100

log_train_image_data = False

cmc_ramp_epochs = 50
cmc_max_weight_2d = 0.075
cmc_max_weight_3d = 0.25

save_epochs = [100]

steps_per_epoch = dataset_samples // batch_size

image_slices = [(0, 480), (160, 640)]
logit_slices = [(0, 120), (40, 160)]

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
    num_classes=40,
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
    num_classes=40,
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
epoch = 250
eval_epoch = 250

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
data_root = "data/NYUv2_MA"
ignore_index = 255

# Name ID TrainID Color
labels = [
    ("background", 0, 0, (128, 0, 0)),
    ("wall", 1, 1, (0, 128, 0)),
    ("floor", 2, 2, (128, 128, 0)),
    ("cabinet", 3, 3, (0, 0, 128)),
    ("bed", 4, 4, (128, 0, 128)),
    ("chair", 5, 5, (0, 128, 128)),
    ("sofa", 6, 6, (128, 128, 128)),
    ("table", 7, 7, (64, 0, 0)),
    ("door", 8, 8, (192, 0, 0)),
    ("window", 9, 9, (64, 128, 0)),
    ("bookshelf", 10, 10, (192, 128, 0)),
    ("picture", 11, 11, (64, 0, 128)),
    ("counter", 12, 12, (192, 0, 128)),
    ("desk", 13, 13, (64, 128, 128)),
    ("shelves", 14, 14, (192, 128, 128)),
    ("curtain", 15, 15, (0, 64, 0)),
    ("dresser", 16, 16, (128, 64, 0)),
    ("pillow", 17, 17, (0, 192, 0)),
    ("mirror", 18, 18, (128, 192, 0)),
    ("floor mat", 19, 19, (0, 64, 128)),
    ("clothes", 20, 20, (128, 64, 128)),
    ("ceiling", 21, 21, (0, 192, 128)),
    ("books", 22, 22, (128, 192, 128)),
    ("refrigerator", 23, 23, (64, 64, 0)),
    ("television", 24, 24, (192, 64, 0)),
    ("paper", 25, 25, (64, 192, 0)),
    ("towel", 26, 26, (192, 192, 0)),
    ("blinds", 27, 27, (64, 64, 128)),
    ("box", 28, 28, (192, 64, 128)),
    ("whiteboard", 29, 29, (64, 192, 128)),
    ("person", 30, 30, (192, 192, 128)),
    ("nightstand", 31, 31, (0, 0, 64)),
    ("toilet", 32, 32, (128, 0, 64)),
    ("sink", 33, 33, (0, 128, 64)),
    ("lamp", 34, 34, (128, 128, 64)),
    ("bathtub", 35, 35, (0, 0, 192)),
    ("bag", 36, 36, (128, 0, 192)),
    ("otherstructure", 37, 37, (0, 128, 192)),
    ("otherfurniture", 38, 38, (128, 128, 192)),
    ("otherprop", 39, 39, (64, 0, 64)),
]

id2label_2d = {label[1]: label for label in labels}
trainId2label_2d = {label[2]: label for label in reversed(labels)}
id2trainId_2d = {label[1]: label[2] for label in labels}
trainId2color_2d = {label[2]: label[3] for label in labels}
class_labels_2d = {label[2]: label[0] for label in reversed(labels)}
colormap_2d = {label[2]: label[3] for label in labels}
id2trainId_3d = id2trainId_2d
class_labels_3d = class_labels_2d
colormap_3d = colormap_2d
id2trainId_original_segment = id2trainId_2d

data = dict(
    num_classes=40,
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
            dict(type="PointClip", point_cloud_range=(-15, -10, -20, 15, 10, 20)),
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
            dict(type="RandCrop", crop_size=(480, 480)),
            dict(type="Blur", student_only=True),
            dict(type="AugMix", student_only=True),
            dict(type="Cutout", p=1, student_only=True),
            dict(type="Cutout", p=0.5, student_only=True),
            dict(type="ExtractFeatures", height=480, width=480),
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
            dict(type="PointClip", point_cloud_range=(-15, -10, -20, 15, 10, 20)),
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
            dict(type="ExtractFeatures", height=480, width=640),
            dict(type="ToTensor"),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
)
