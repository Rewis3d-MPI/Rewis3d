"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class CriteriaSimple(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss


class CriteriaTeacherStudent(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(
        self,
        pred_student,
        pred_teacher,
        masks_student,
        teacher_weight,
        teacher_hard_label,
    ):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred_student
        loss = 0
        for c in self.criteria:
            loss += c(
                pred_student,
                pred_teacher,
                masks_student,
                teacher_weight,
                teacher_hard_label,
            )
        return loss


class CriteriaMain(object):
    def __init__(self, criteria=None):
        self.criteria = LOSSES.build(cfg=criteria)

    def __call__(
        self, cfg, model_3d, model_2d, input_dict, epoch, loss_fn_3d, loss_fn_2d
    ):
        return self.criteria(
            cfg, model_3d, model_2d, input_dict, epoch, loss_fn_3d, loss_fn_2d
        )


def build_criteria(cfg, model_class="simple"):
    if "simple" in model_class:
        return CriteriaSimple(cfg)
    elif "student_teacher" in model_class:
        return CriteriaTeacherStudent(cfg)
    elif "main_criteria" in model_class:
        return CriteriaMain(cfg)
