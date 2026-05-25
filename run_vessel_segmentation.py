"""
Simple script to run vessel segmentation and ROI extraction
Usage: python run_vessel_segmentation.py --input patient_0000.nii.gz --output results/
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.my_utils.vessel_segmentation import VesselSegmentationPredictor
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Run vessel segmentation and ROI extraction')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input NIfTI file (e.g., patient_0000.nii.gz)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save-roi', action='store_true',
                        help='Save ROI volume as numpy array')
    parser.add_argument('--save-seg', action='store_true',
                        help='Save segmentation mask as numpy array')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("🧠 Brain Vessel Segmentation & ROI Extraction")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output dir: {args.output}")
    print(f"Device: {args.device}")
    print()
    
    # Model paths (adjust if needed)
    model_base = PROJECT_ROOT
    
    print("📦 Loading models...")
    print("  - Sparse search model (coarse localization)")
    print("  - Fine segmentation model (primary)")
    print("  - Fine segmentation model (recall-focused)")
    print()
    
    try:
        # Initialize predictor
        predictor = VesselSegmentationPredictor(
            # Primary fine segmentation model
            model_path=str(model_base / "nnunet-da3-sklr-ep800"),
            
            # Additional fine segmentation model
            additional_dense_model_paths=[
                str(model_base / "nnunet-da6-sklr-w3-tv07")
            ],
            
            # Sparse search model
            sparse_model_path=str(model_base / "nnunet-vessel-grouping-da7"),
            
            folds=["all"],
            device=args.device,
            use_sparse_search=True,
            verbose=True
        )
        
        print("\n✅ Models loaded successfully!")
        print()
        
        # Run inference
        print("🔄 Running vessel segmentation...")
        seg, roi, transform_info = predictor.predict_from_file(
            file_path=args.input,
            return_probabilities=False
        )
        
        print("\n✅ Segmentation complete!")
        print()
        
        # Print results
        print("📊 Results:")
        print(f"  Segmentation shape: {seg.shape}")
        print(f"  ROI shape: {roi.shape}")
        print(f"  ROI dtype: {roi.dtype}")
        print()
        
        # Save results
        base_name = Path(args.input).stem.replace('.nii', '').replace('_0000', '')
        
        if args.save_seg:
            seg_path = os.path.join(args.output, f"{base_name}_segmentation.npy")
            seg_np = seg.cpu().numpy() if torch.is_tensor(seg) else seg
            np.save(seg_path, seg_np)
            print(f"💾 Saved segmentation: {seg_path}")
        
        if args.save_roi:
            roi_path = os.path.join(args.output, f"{base_name}_roi.npy")
            roi_np = roi.cpu().numpy() if torch.is_tensor(roi) else roi
            np.save(roi_path, roi_np)
            print(f"💾 Saved ROI: {roi_path}")
        
        # Save transform info
        import json
        transform_path = os.path.join(args.output, f"{base_name}_transform.json")
        with open(transform_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_info = convert_to_serializable(transform_info)
            json.dump(serializable_info, f, indent=2)
        print(f"💾 Saved transform info: {transform_path}")
        
        print()
        print("=" * 60)
        print("✅ Processing complete!")
        print("=" * 60)
        print()
        print("📝 Next steps:")
        print("  1. Use the ROI volume for aneurysm classification")
        print("  2. Use the segmentation mask for vessel analysis")
        print("  3. Use transform_info for coordinate mapping")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
