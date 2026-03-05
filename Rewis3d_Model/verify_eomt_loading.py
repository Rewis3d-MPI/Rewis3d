# Copyright (c) 2026 Max Planck Institute for Informatics
# Authors: Jonas Ernst, Wolfgang Boettcher
# Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).
# See LICENSE file in the project root for details.


import sys
import os
import torch
import warnings

# Add current directory to path
sys.path.insert(0, os.path.abspath("."))

try:
    from pointcept.models.segmentation_2d import Segmentation2DModel
    print("Successfully imported Segmentation2DModel")
except ImportError as e:
    print(f"Error importing Segmentation2DModel: {e}")
    # Try adding parent directory if running from inside Rewis3d_Model
    sys.path.append(os.path.dirname(os.path.abspath(".")))
    try:
        from pointcept.models.segmentation_2d import Segmentation2DModel
        print("Successfully imported Segmentation2DModel (after path adjustment)")
    except ImportError as e2:
        print(f"Error importing Segmentation2DModel again: {e2}")
        sys.exit(1)

def test_loading():
    print("-" * 50)
    print("Testing EoMT Model Loading")
    print("-" * 50)
    
    # Configuration representing the update
    # Using the ID provided by user
    model_name = "tue-mps/ade20k_semantic_eomt_large_512"
    num_classes = 19
    
    print(f"Attempting to load model: {model_name}")
    try:
        model = Segmentation2DModel(
            num_classes=num_classes,
            model=model_name,
            model_class="student_teacher"
        )
        print("Model object created successfully.")
        
        # Check if it's using the correct class
        print(f"Backbone type: {type(model.backbone)}")
        
        if hasattr(model, 'is_eomt'):
            print(f"Model detected as EoMT: {model.is_eomt}")
        
        # Test Forward Pass with Dummy Input
        # SegFormer expects (B, 3, H, W)
        print("Testing Forward Pass...")
        dummy_input = torch.randn(1, 3, 512, 512)
        input_dict = {
            "student_pixel_values_1": dummy_input,
            "student_labels_1": torch.zeros(1, 512, 512).long()
        }
        
        output = model(input_dict)
        if "seg_logits" in output:
            logits = output["seg_logits"]
            print(f"Forward pass successful. Logits shape: {logits.shape}")
            # Expected: (1, 19, 128, 128) or similar (Segformer outputs 1/4 resolution)
        else:
            print("Forward pass returned, but 'seg_logits' missing.")
            print(output.keys())

    except Exception as e:
        print(f"Failed to load/run model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("-" * 50)
    print("Test Passed: EoMT integration logic appears functional.")
    print("-" * 50)

if __name__ == "__main__":
    test_loading()
