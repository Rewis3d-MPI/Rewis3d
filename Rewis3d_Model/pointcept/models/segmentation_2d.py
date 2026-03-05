import torch.nn as nn

from transformers import SegformerForSemanticSegmentation, AutoModelForSemanticSegmentation

from .builder import MODELS


@MODELS.register_module()
class Segmentation2DModel(nn.Module):
    def __init__(
        self,
        num_classes,
        model,
        model_class="simple",
        **kwargs,
    ):
        super().__init__()
        self.is_eomt = "eomt" in model.lower()
        if self.is_eomt:
            try:
                from transformers import EomtForUniversalSegmentation
            except ImportError:
                # Fallback or raise error if transformers implies it should work
                # Trying dynamic import or AutoModel as fallback if class unavailable
                from transformers import AutoModelForUniversalSegmentation as EomtForUniversalSegmentation

            self.backbone = EomtForUniversalSegmentation.from_pretrained(
                model,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            # EoMT might need mapping number of classes if not handled by from_pretrained
        else:
            self.backbone = SegformerForSemanticSegmentation.from_pretrained(
                model,
                return_dict=True,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        self.mode = "student_"
        self.model_class = model_class

    def set_to_teacher(self):
        self.mode = "teacher_"

    def forward(self, input_dict):
        pixel_values = input_dict[self.mode + "pixel_values_1"]
        labels = input_dict["student_labels_1"]

        if len(pixel_values.shape) != 4:
            pixel_values = pixel_values.unsqueeze(0)
            labels = labels.unsqueeze(0)

        if self.is_eomt:
             # EoMT Forward Pass
             # output_hidden_states=True might not be supported or needed for logits
            outputs = self.backbone(pixel_values=pixel_values)
            
            # Universal/Mask Transformer Output Processing
            # Assuming outputs has masks_queries_logits and class_queries_logits
            # mask_logits: (B, Q, H/4, W/4) usually
            # class_logits: (B, Q, num_classes + 1)
            
            mask_logits = outputs.masks_queries_logits
            class_logits = outputs.class_queries_logits
            
            # Convert to semantic logits: (B, C, H, W)
            # prob = sigmoid(mask) * softmax(class)
            # We want logits, but approximation via probability sum is standard for inference.
            # For training, we need differentiable path.
            
            # Softmax over classes
            pred_class = class_logits.softmax(dim=-1)
            # Exclude 'no object' class (last one usually)
            pred_class = pred_class[:, :, :-1] 
            
            pred_mask = mask_logits.sigmoid()
            
            # (B, Q, C) @ (B, Q, H_mask, W_mask) -> (B, C, H_mask, W_mask)
            # einsum: bqc, bqhw -> bchw
            sem_probs = torch.einsum("bqc, bqhw -> bchw", pred_class, pred_mask)
            
            # Convert probs back to logits/pseudo-logits for consistency (or usage as probs)
            # Since loss (CrossEntropy) expects logits, we can take log(sem_probs + epsilon)
            # or just use sem_probs if using a different loss. 
            # However, Segformer branch returns LOGITS.
            # To emulate logits:
            logits = torch.log(sem_probs + 1e-6)
            
        else:
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
            logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        output_dict = dict(seg_logits=logits)
        output_dict[self.mode + "upsampled_logits_2d"] = upsampled_logits

        return output_dict
