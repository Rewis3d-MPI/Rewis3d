import torch.nn as nn
from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class Segmentation3DModel(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        model_class="simple",
        **kwargs,
    ):
        super().__init__()

        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

        self.mode = "student_"
        self.model_class = model_class

    def set_to_teacher(self):
        self.mode = "teacher_"

    def forward(self, input_dict):
        if self.mode == "teacher_":
            input_dict["coord"] = input_dict["teacher_coord"]
            input_dict["feat"] = input_dict["teacher_feat"]
        else:
            input_dict["coord"] = input_dict["student_coord"]
            input_dict["feat"] = input_dict["student_feat"]
        input_dict["offset"] = input_dict["student_offset"]
        input_dict["segment"] = input_dict["student_segment"]
        if "student_colors" in input_dict:
            input_dict["color"] = input_dict["student_colors"]

        input_dict["grid_size"] = 0.05
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)

        output_dict = dict(seg_logits=seg_logits)

        return output_dict
