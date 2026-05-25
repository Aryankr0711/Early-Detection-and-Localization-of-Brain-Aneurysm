#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
"""
pipeline.py — End-to-End RSNA 2025 Brain Aneurysm Detection Pipeline
=====================================================================

Combines the full preprocessing pipeline and the vessel segmentation + ROI
extraction pipeline from the RSNA 2025 1st place solution.

Usage:
    python pipeline.py --patient-path /path/to/dicom/folder
    python pipeline.py --patient-path /path/to/dicom/folder --device cpu --save-roi --save-seg

Pipeline stages:
    1. DICOM auditing (audit_folder)
    2. Majority-based slice filtering (prepare_majority_subset) with slice spacing filtering
    3. DICOM → NIfTI conversion (dcm2niix + gdcmconv fallback)
    4. Volume loading with RAS orientation standardisation (SimpleITKIOWithReorient)
    5. Per-volume z-score normalisation (handled internally by nnU-Net preprocessing)
    6. Vessel segmentation (VesselSegmentationPredictor with sparse + dense models)
    7. ROI extraction from segmentation output

All functions, modules, and logic are strictly reused from the original codebase.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 1. ENVIRONMENT SETUP — Detect Kaggle vs local, resolve PROJECT_ROOT
# ============================================================================

_IS_KAGGLE = os.path.exists("/kaggle")

# Resolve project root from this file's location (pipeline.py lives at repo root)
PROJECT_ROOT = Path(__file__).resolve().parent

# Insert project root *and* the custom nnUNet into sys.path BEFORE any repo imports.
# This makes the `rootutils.setup_root()` calls inside the source modules no-ops
# (the paths are already configured).
_paths_to_add = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "nnUNet"),
]
for _p in reversed(_paths_to_add):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Default output directory (writable on Kaggle)
if _IS_KAGGLE:
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/pipeline_outputs")
else:
    DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline_outputs"

# Default model directories (relative to PROJECT_ROOT)
DEFAULT_PRIMARY_MODEL = str(PROJECT_ROOT / "nnunet-da3-sklr-ep800")
DEFAULT_SPARSE_MODEL = str(PROJECT_ROOT / "nnunet-vessel-grouping-da7")
DEFAULT_ADDITIONAL_DENSE_MODEL = str(PROJECT_ROOT / "nnunet-da6-sklr-w3-tv07")

# ============================================================================
# 2. IMPORTS — All from the original codebase
# ============================================================================

import torch
import nibabel as nib

# Preprocessing imports (from src.my_utils.rsna_dcm2niix)
from src.my_utils.rsna_dcm2niix import (
    audit_folder,
    prepare_majority_subset,
    convert_dicom_to_nifti,
    which_or_die,
)

# Vessel segmentation imports
from src.my_utils.vessel_segmentation import (
    VesselSegmentationPredictor,
    VesselSegmentationOutput,
    VesselSegmentationStage,
)

# nnU-Net I/O (used inside VesselSegmentationPredictor, also available for direct use)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIOWithReorient

# nnU-Net normalisation (referenced for documentation; actual normalisation
# happens inside nnU-Net's _preprocess_data during inference)
from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
    ZScoreNormalization,
)


# ============================================================================
# 3. STRUCTURED LOGGING HELPERS
# ============================================================================

def _log_stage(stage_name: str, msg: str) -> None:
    """Print a structured log message for a pipeline stage."""
    print(f"[{stage_name}] {msg}")


def _log_separator(title: str) -> None:
    """Print a visual separator."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _tensor_stats(t) -> Dict[str, Any]:
    """Compute basic statistics for a tensor or numpy array."""
    if t is None:
        return {"value": None}
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().float().numpy()
    else:
        arr = np.asarray(t, dtype=np.float32)
    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "has_nan": bool(np.any(np.isnan(arr))),
        "has_inf": bool(np.any(np.isinf(arr))),
    }
    return stats


# ============================================================================
# 4. PIPELINE CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    patient_path: str = ""
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cuda"

    # Model paths (configurable)
    primary_model_path: str = DEFAULT_PRIMARY_MODEL
    sparse_model_path: str = DEFAULT_SPARSE_MODEL
    additional_dense_model_path: str = DEFAULT_ADDITIONAL_DENSE_MODEL

    # Vessel segmentation parameters (matching rsna_submission_roi.py)
    folds: List[str] = field(default_factory=lambda: ["all"])
    use_sparse_search: bool = True
    use_mirroring: bool = False
    enable_orientation_correction: bool = True
    verbose: bool = True

    # Save options
    save_roi: bool = False
    save_seg: bool = False


# ============================================================================
# 5. run_preprocessing() — DICOM → NIfTI
# ============================================================================

def run_preprocessing(
    patient_path: str,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    verbose: bool = True,
) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Execute the complete DICOM-to-NIfTI preprocessing pipeline.

    Stages:
        1. Input validation
        2. DICOM auditing (audit_folder)
        3. Majority-based slice filtering (prepare_majority_subset)
        4. DICOM→NIfTI conversion (convert_dicom_to_nifti with dcm2niix + gdcmconv fallback)
        5. Output validation

    Args:
        patient_path: Path to patient DICOM folder (single series)
        output_dir: Directory for NIfTI output
        verbose: Enable detailed logging

    Returns:
        (nifti_path or None, log_messages, error_messages)
    """
    _log_separator("STAGE 1: PREPROCESSING (DICOM → NIfTI)")
    t0 = time.perf_counter()
    all_logs: List[str] = []
    all_errors: List[str] = []

    dir_path = Path(patient_path)
    out_dir = Path(output_dir) / "nifti"

    # ── 1. Input validation ──────────────────────────────────────────────
    _log_stage("PREPROCESS", "Validating input...")
    if not dir_path.exists():
        msg = f"Patient path does not exist: {dir_path}"
        _log_stage("PREPROCESS", f"❌ {msg}")
        all_errors.append(msg)
        return None, all_logs, all_errors

    if not dir_path.is_dir():
        msg = f"Patient path is not a directory: {dir_path}"
        _log_stage("PREPROCESS", f"❌ {msg}")
        all_errors.append(msg)
        return None, all_logs, all_errors

    # Check that dcm2niix and gdcmconv are available
    try:
        dcm2niix_path = which_or_die("dcm2niix")
        gdcmconv_path = which_or_die("gdcmconv")
        _log_stage("PREPROCESS", f"dcm2niix found: {dcm2niix_path}")
        _log_stage("PREPROCESS", f"gdcmconv found: {gdcmconv_path}")
    except FileNotFoundError as e:
        msg = str(e)
        _log_stage("PREPROCESS", f"❌ {msg}")
        if _IS_KAGGLE:
            _log_stage("PREPROCESS", "Tip: Install with: !apt-get install -y dcm2niix libgdcm-tools")
        all_errors.append(msg)
        return None, all_logs, all_errors

    _log_stage("PREPROCESS", f"Input directory: {dir_path}")
    _log_stage("PREPROCESS", f"Output directory: {out_dir}")

    # ── 2. DICOM auditing ────────────────────────────────────────────────
    _log_stage("PREPROCESS", "Running DICOM audit...")
    file_exts = (".dcm", "")
    audit = audit_folder(dir_path, file_exts)
    _log_stage("PREPROCESS", f"  Total files:  {audit.total_files}")
    _log_stage("PREPROCESS", f"  DICOM files:  {audit.dicom_files}")
    _log_stage("PREPROCESS", f"  Majority key: {audit.majority_key}")
    _log_stage("PREPROCESS", f"  Series count: {len(audit.series_map)}")
    all_logs.append(
        f"Audit: dicom={audit.dicom_files}, majority={audit.majority_key}, "
        f"series={len(audit.series_map)}"
    )

    if audit.dicom_files == 0:
        msg = f"No DICOM files found in {dir_path}"
        _log_stage("PREPROCESS", f"❌ {msg}")
        all_errors.append(msg)
        return None, all_logs, all_errors

    # ── 3-4. DICOM → NIfTI conversion ────────────────────────────────────
    # Uses convert_dicom_to_nifti() which internally handles:
    #   - Majority-based subset preparation (prepare_majority_subset)
    #   - Slice spacing filtering
    #   - dcm2niix conversion with gdcmconv fallback
    #   - Primary NIfTI file selection
    #   - 3D normalisation (force_3d_nii)
    #
    # Flags match rsna_submission_roi.py exactly:
    #   - "-z", "n" : uncompressed NIfTI (fast I/O)
    #   - "-b", "y" : generate JSON sidecar
    #   - "-i", "n" : allow DERIVED/MPR
    #   - "-f", "%s": filename = SeriesInstanceUID
    #   - convert_to_ras=False : RAS handled later by SimpleITKIOWithReorient
    _log_stage("PREPROCESS", "Converting DICOM → NIfTI...")
    tmp_root = Path(tempfile.gettempdir())

    nifti_path, used_dir, logs, errors = convert_dicom_to_nifti(
        dir_path=dir_path,
        out_dir=out_dir,
        tmp_root=tmp_root,
        use_majority_size=True,
        copy_mode="auto",
        file_exts=file_exts,
        dcm2niix_flags=("-z", "n", "-b", "y", "-i", "n", "-f", "%s"),
        gdcm_first=False,
        use_slice_spacing_filter=True,
        slice_spacing_tolerance=2.0,
        convert_to_ras=False,  # RAS handled by SimpleITKIOWithReorient during loading
    )

    all_logs.extend(logs)
    all_errors.extend(errors)

    if verbose and logs:
        _log_stage("PREPROCESS", "Conversion logs:")
        for line in logs:
            _log_stage("PREPROCESS", f"  {line}")
    if errors:
        _log_stage("PREPROCESS", "Conversion errors:")
        for line in errors:
            _log_stage("PREPROCESS", f"  ⚠ {line}")

    # ── 5. Output validation ─────────────────────────────────────────────
    if nifti_path is None:
        msg = "NIfTI conversion failed — no output file generated."
        _log_stage("PREPROCESS", f"❌ {msg}")
        all_errors.append(msg)
        return None, all_logs, all_errors

    nifti_path_str = str(nifti_path)
    if not Path(nifti_path_str).exists():
        msg = f"NIfTI file does not exist after conversion: {nifti_path_str}"
        _log_stage("PREPROCESS", f"❌ {msg}")
        all_errors.append(msg)
        return None, all_logs, all_errors

    # Quick validation: load with nibabel and check shape
    try:
        nii = nib.load(nifti_path_str)
        shape = nii.shape
        spacing = nii.header.get_zooms()[:3]
        _log_stage("PREPROCESS", f"✅ NIfTI generated successfully: {nifti_path_str}")
        _log_stage("PREPROCESS", f"   Shape:   {shape}")
        _log_stage("PREPROCESS", f"   Spacing: {spacing}")
        _log_stage("PREPROCESS", f"   Dtype:   {nii.header.get_data_dtype()}")

        if len(shape) < 3:
            msg = f"NIfTI has unexpected dimensionality: {shape}"
            _log_stage("PREPROCESS", f"⚠ {msg}")
            all_errors.append(msg)
    except Exception as e:
        msg = f"Failed to validate NIfTI: {e}"
        _log_stage("PREPROCESS", f"⚠ {msg}")
        all_errors.append(msg)

    dt = time.perf_counter() - t0
    _log_stage("PREPROCESS", f"Preprocessing completed in {dt:.2f}s")

    return nifti_path_str, all_logs, all_errors


# ============================================================================
# 6. run_vessel_segmentation() — Vessel Segmentation + ROI Extraction
# ============================================================================

def run_vessel_segmentation(
    nifti_path: str,
    config: Optional[PipelineConfig] = None,
) -> VesselSegmentationOutput:
    """
    Execute vessel segmentation and ROI extraction on a NIfTI volume.

    Uses the exact same VesselSegmentationPredictor configuration as the
    winner's rsna_submission_roi.py:
        - Primary model: nnunet-da3-sklr-ep800
        - Additional dense model: nnunet-da6-sklr-w3-tv07
        - Sparse search model: nnunet-vessel-grouping-da7
        - Folds: ["all"]
        - Sparse search enabled
        - Orientation correction enabled
        - Mirroring (TTA) disabled

    Internally, nnU-Net handles:
        - Volume loading via SimpleITKIOWithReorient (RAS orientation)
        - Per-volume z-score normalisation (ZScoreNormalization)
        - Preprocessing (resampling to network resolution)
        - Sparse search for coarse localisation
        - Dense inference on ROI
        - ROI re-extraction with refinement

    Args:
        nifti_path: Path to NIfTI file
        config: Pipeline configuration (uses defaults if None)

    Returns:
        VesselSegmentationOutput containing seg, roi, transform_info
    """
    _log_separator("STAGE 2: VESSEL SEGMENTATION + ROI EXTRACTION")
    t0 = time.perf_counter()

    if config is None:
        config = PipelineConfig()

    # ── 1. Input validation ──────────────────────────────────────────────
    _log_stage("VESSEL_SEG", "Validating input...")
    if not Path(nifti_path).exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    # Validate model directories exist
    for name, path in [
        ("Primary model", config.primary_model_path),
        ("Sparse model", config.sparse_model_path),
        ("Additional dense model", config.additional_dense_model_path),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} directory not found: {path}")
        _log_stage("VESSEL_SEG", f"  {name}: {path}")

    _log_stage("VESSEL_SEG", f"  Device: {config.device}")
    _log_stage("VESSEL_SEG", f"  Folds: {config.folds}")
    _log_stage("VESSEL_SEG", f"  Sparse search: {config.use_sparse_search}")
    _log_stage("VESSEL_SEG", f"  Orientation correction: {config.enable_orientation_correction}")

    # ── 2. Load volume via SimpleITKIOWithReorient (RAS) ─────────────────
    _log_stage("VESSEL_SEG", "Loading NIfTI volume with RAS orientation...")
    io_handler = SimpleITKIOWithReorient()
    image, properties = io_handler.read_images([nifti_path], orientation="RAS")

    _log_stage("VESSEL_SEG", f"  Volume shape (C,Z,Y,X): {image.shape}")
    _log_stage("VESSEL_SEG", f"  Spacing (z,y,x): {properties.get('spacing', 'N/A')}")
    _log_stage("VESSEL_SEG", f"  Dtype: {image.dtype}")

    # Validate loaded volume
    if image.ndim < 4:
        raise ValueError(f"Expected 4D volume (C,Z,Y,X), got ndim={image.ndim}")
    if np.any(np.isnan(image)):
        _log_stage("VESSEL_SEG", "⚠ WARNING: Volume contains NaN values")
    if np.any(np.isinf(image)):
        _log_stage("VESSEL_SEG", "⚠ WARNING: Volume contains Inf values")

    # ── 3. Initialise VesselSegmentationPredictor ────────────────────────
    # Configuration matches rsna_submission_roi.py and run_vessel_segmentation.py
    _log_stage("VESSEL_SEG", "Initialising VesselSegmentationPredictor...")
    _log_stage("VESSEL_SEG", "  Loading models (this may take a moment)...")

    predictor = VesselSegmentationPredictor(
        # Primary fine segmentation model
        model_path=config.primary_model_path,
        # Additional fine segmentation model (recall-focused)
        additional_dense_model_paths=[config.additional_dense_model_path],
        # Sparse search model (coarse localisation)
        sparse_model_path=config.sparse_model_path,
        # Fold configuration
        folds=config.folds,
        # Device
        device=config.device,
        # Sparse search: enabled for efficient inference
        use_sparse_search=config.use_sparse_search,
        # Mirroring (TTA): disabled matching submission
        use_mirroring=config.use_mirroring,
        # Orientation correction: enabled (requires sparse model with 4 classes)
        enable_orientation_correction=config.enable_orientation_correction,
        # Verbose output
        verbose=config.verbose,
    )

    _log_stage("VESSEL_SEG", "✅ Models loaded successfully.")

    # ── 4. Run inference ─────────────────────────────────────────────────
    # Uses predict_single_volume_with_info() — the same method called by
    # RsnaRoiPipeline.vessel_predict() in rsna_submission_roi.py
    _log_stage("VESSEL_SEG", "Running vessel segmentation inference...")

    result: VesselSegmentationOutput = predictor.predict_single_volume_with_info(
        image=image,
        properties=properties,
        return_probabilities=False,
        stage_limit=VesselSegmentationStage.AGGREGATE,
    )

    # ── 5. Validate output ───────────────────────────────────────────────
    _log_stage("VESSEL_SEG", "Validating segmentation output...")

    if result.seg is not None:
        seg_stats = _tensor_stats(result.seg)
        _log_stage("VESSEL_SEG", f"  Segmentation shape: {seg_stats['shape']}")
        _log_stage("VESSEL_SEG", f"  Segmentation dtype:  {seg_stats['dtype']}")
        _log_stage("VESSEL_SEG", f"  Segmentation range:  [{seg_stats['min']:.4f}, {seg_stats['max']:.4f}]")
        if seg_stats["has_nan"]:
            _log_stage("VESSEL_SEG", "  ⚠ WARNING: Segmentation contains NaN")
        if seg_stats["has_inf"]:
            _log_stage("VESSEL_SEG", "  ⚠ WARNING: Segmentation contains Inf")
    else:
        _log_stage("VESSEL_SEG", "  ⚠ WARNING: Segmentation is None")

    if result.roi is not None:
        roi_stats = _tensor_stats(result.roi)
        _log_stage("VESSEL_SEG", f"  ROI shape: {roi_stats['shape']}")
        _log_stage("VESSEL_SEG", f"  ROI dtype:  {roi_stats['dtype']}")
        _log_stage("VESSEL_SEG", f"  ROI range:  [{roi_stats['min']:.4f}, {roi_stats['max']:.4f}]")
        _log_stage("VESSEL_SEG", f"  ROI mean:   {roi_stats['mean']:.4f}")
        _log_stage("VESSEL_SEG", f"  ROI std:    {roi_stats['std']:.4f}")
        if roi_stats["has_nan"]:
            _log_stage("VESSEL_SEG", "  ⚠ WARNING: ROI contains NaN")
        if roi_stats["has_inf"]:
            _log_stage("VESSEL_SEG", "  ⚠ WARNING: ROI contains Inf")
    else:
        _log_stage("VESSEL_SEG", "  ⚠ WARNING: ROI is None")

    _log_stage("VESSEL_SEG", f"  Completed stage: {result.completed_stage}")
    _log_stage("VESSEL_SEG", f"  Stage history: {[s.value for s in result.stage_history]}")
    _log_stage("VESSEL_SEG", f"  Is full pipeline: {result.is_full}")

    dt = time.perf_counter() - t0
    _log_stage("VESSEL_SEG", f"✅ Vessel segmentation completed in {dt:.2f}s")

    # Free loaded volume from memory
    del image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# 7. run_roi_extraction() — Extract and validate the ROI volume
# ============================================================================

def run_roi_extraction(
    vessel_output: VesselSegmentationOutput,
    output_dir: Optional[str] = None,
    save_roi: bool = False,
    save_seg: bool = False,
) -> Dict[str, Any]:
    """
    Extract and validate the ROI volume from vessel segmentation output.

    In the winner's pipeline, ROI extraction happens inside
    VesselSegmentationPredictor._run_dense_inference():
        roi_data = (roi_data - roi_data.mean()) / (roi_data.std() + 1e-6)
    The ROI is the z-score normalised image crop corresponding to the
    detected vessel region. This function validates and optionally saves it.

    Args:
        vessel_output: VesselSegmentationOutput from run_vessel_segmentation()
        output_dir: Directory to save outputs (if save_roi or save_seg)
        save_roi: Whether to save the ROI volume to disk
        save_seg: Whether to save the segmentation mask to disk

    Returns:
        Dictionary with:
            - "roi_volume": numpy array (C, Z, Y, X)
            - "roi_shape": tuple
            - "roi_spacing": spacing info from transform_info (if available)
            - "roi_stats": basic statistics
            - "seg_volume": numpy array (if available)
            - "transform_info": dict with coordinate transform metadata
            - "saved_files": list of saved file paths
    """
    _log_separator("STAGE 3: ROI EXTRACTION & VALIDATION")
    t0 = time.perf_counter()

    saved_files: List[str] = []
    results: Dict[str, Any] = {
        "roi_volume": None,
        "roi_shape": None,
        "roi_spacing": None,
        "roi_stats": None,
        "seg_volume": None,
        "transform_info": None,
        "saved_files": saved_files,
    }

    # ── 1. Extract ROI ───────────────────────────────────────────────────
    if vessel_output.roi is None:
        _log_stage("ROI_EXTRACT", "❌ ROI volume is None — segmentation may have failed.")
        return results

    roi_tensor = vessel_output.roi
    if isinstance(roi_tensor, torch.Tensor):
        roi_np = roi_tensor.detach().cpu().numpy()
    else:
        roi_np = np.asarray(roi_tensor)

    results["roi_volume"] = roi_np
    results["roi_shape"] = tuple(roi_np.shape)
    results["transform_info"] = vessel_output.transform_info

    # ── 2. Extract spacing from transform_info ───────────────────────────
    spacing_info = None
    if vessel_output.transform_info and isinstance(vessel_output.transform_info, dict):
        # The transform_info contains spacing under various keys
        spacing_info = vessel_output.transform_info.get("spacing_after_resampling")
        if spacing_info is None:
            spacing_info = vessel_output.transform_info.get("spacing")
        results["roi_spacing"] = spacing_info

    # ── 3. Compute statistics ────────────────────────────────────────────
    roi_stats = _tensor_stats(roi_np)
    results["roi_stats"] = roi_stats

    _log_stage("ROI_EXTRACT", "ROI Volume Summary:")
    _log_stage("ROI_EXTRACT", f"  Shape (C,Z,Y,X): {roi_stats['shape']}")
    _log_stage("ROI_EXTRACT", f"  Dtype:           {roi_stats['dtype']}")
    _log_stage("ROI_EXTRACT", f"  Min:             {roi_stats['min']:.6f}")
    _log_stage("ROI_EXTRACT", f"  Max:             {roi_stats['max']:.6f}")
    _log_stage("ROI_EXTRACT", f"  Mean:            {roi_stats['mean']:.6f}")
    _log_stage("ROI_EXTRACT", f"  Std:             {roi_stats['std']:.6f}")
    _log_stage("ROI_EXTRACT", f"  Has NaN:         {roi_stats['has_nan']}")
    _log_stage("ROI_EXTRACT", f"  Has Inf:         {roi_stats['has_inf']}")
    if spacing_info:
        _log_stage("ROI_EXTRACT", f"  Spacing:         {spacing_info}")

    # Validation warnings
    if roi_stats["has_nan"]:
        _log_stage("ROI_EXTRACT", "⚠ WARNING: ROI contains NaN values — downstream classification may fail")
    if roi_stats["has_inf"]:
        _log_stage("ROI_EXTRACT", "⚠ WARNING: ROI contains Inf values — downstream classification may fail")
    if roi_np.size == 0:
        _log_stage("ROI_EXTRACT", "⚠ WARNING: ROI volume is empty (0 voxels)")

    # ── 4. Extract segmentation ──────────────────────────────────────────
    if vessel_output.seg is not None:
        seg_tensor = vessel_output.seg
        if isinstance(seg_tensor, torch.Tensor):
            seg_np = seg_tensor.detach().cpu().numpy()
        else:
            seg_np = np.asarray(seg_tensor)
        results["seg_volume"] = seg_np

        # Segmentation label statistics
        unique_labels = np.unique(seg_np)
        _log_stage("ROI_EXTRACT", f"  Segmentation shape:  {seg_np.shape}")
        _log_stage("ROI_EXTRACT", f"  Unique labels:       {unique_labels.tolist()}")
        for label in unique_labels:
            count = int(np.sum(seg_np == label))
            _log_stage("ROI_EXTRACT", f"    Label {int(label)}: {count} voxels")

    # ── 5. Save to disk (optional) ───────────────────────────────────────
    if (save_roi or save_seg) and output_dir:
        save_dir = Path(output_dir) / "results"
        save_dir.mkdir(parents=True, exist_ok=True)

        if save_roi:
            roi_path = save_dir / "roi_volume.npy"
            np.save(str(roi_path), roi_np.astype(np.float16, copy=False))
            saved_files.append(str(roi_path))
            _log_stage("ROI_EXTRACT", f"💾 Saved ROI volume: {roi_path}")

        if save_seg and vessel_output.seg is not None:
            seg_path = save_dir / "segmentation.npy"
            np.save(str(seg_path), seg_np.astype(np.uint8, copy=False))
            saved_files.append(str(seg_path))
            _log_stage("ROI_EXTRACT", f"💾 Saved segmentation: {seg_path}")

        if vessel_output.transform_info:
            xform_path = save_dir / "transform_info.json"
            # Serialise numpy types
            def _to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: _to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [_to_serializable(item) for item in obj]
                return obj

            with open(str(xform_path), "w", encoding="utf-8") as f:
                json.dump(_to_serializable(vessel_output.transform_info), f, indent=2)
            saved_files.append(str(xform_path))
            _log_stage("ROI_EXTRACT", f"💾 Saved transform info: {xform_path}")

    dt = time.perf_counter() - t0
    _log_stage("ROI_EXTRACT", f"✅ ROI extraction completed in {dt:.2f}s")

    return results


# ============================================================================
# 8. run_full_pipeline() — Orchestrator
# ============================================================================

def run_full_pipeline(
    patient_path: str,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Execute the complete end-to-end pipeline:
        patient_path → preprocessing → vessel_segmentation → roi_extraction

    Args:
        patient_path: Path to patient DICOM folder (single series)
        config: Pipeline configuration (uses defaults if None)

    Returns:
        Dictionary with all results from each stage:
            - "nifti_path": str or None
            - "preprocessing_logs": list
            - "preprocessing_errors": list
            - "vessel_output": VesselSegmentationOutput
            - "roi_results": dict from run_roi_extraction()
            - "total_time": float (seconds)
            - "success": bool
    """
    if config is None:
        config = PipelineConfig(patient_path=patient_path)

    _log_separator("RSNA 2025 — End-to-End Pipeline")
    print(f"  Patient path:  {patient_path}")
    print(f"  Output dir:    {config.output_dir}")
    print(f"  Device:        {config.device}")
    print(f"  Save ROI:      {config.save_roi}")
    print(f"  Save SEG:      {config.save_seg}")

    pipeline_t0 = time.perf_counter()
    pipeline_result: Dict[str, Any] = {
        "nifti_path": None,
        "preprocessing_logs": [],
        "preprocessing_errors": [],
        "vessel_output": None,
        "roi_results": None,
        "total_time": 0.0,
        "success": False,
    }

    # ── Stage 1: Preprocessing ───────────────────────────────────────────
    try:
        nifti_path, preprocess_logs, preprocess_errors = run_preprocessing(
            patient_path=patient_path,
            output_dir=config.output_dir,
            verbose=config.verbose,
        )
        pipeline_result["nifti_path"] = nifti_path
        pipeline_result["preprocessing_logs"] = preprocess_logs
        pipeline_result["preprocessing_errors"] = preprocess_errors

        if nifti_path is None:
            _log_stage("PIPELINE", "❌ Preprocessing failed — aborting pipeline.")
            pipeline_result["total_time"] = time.perf_counter() - pipeline_t0
            return pipeline_result

    except Exception as e:
        _log_stage("PIPELINE", f"❌ Preprocessing exception: {e}")
        import traceback
        traceback.print_exc()
        pipeline_result["preprocessing_errors"].append(str(e))
        pipeline_result["total_time"] = time.perf_counter() - pipeline_t0
        return pipeline_result

    # ── Stage 2: Vessel Segmentation ─────────────────────────────────────
    try:
        vessel_output = run_vessel_segmentation(
            nifti_path=nifti_path,
            config=config,
        )
        pipeline_result["vessel_output"] = vessel_output

        if not vessel_output.is_full:
            _log_stage("PIPELINE", "⚠ Vessel segmentation did not complete all stages.")
            if vessel_output.roi is None:
                _log_stage("PIPELINE", "❌ No ROI produced — aborting pipeline.")
                pipeline_result["total_time"] = time.perf_counter() - pipeline_t0
                return pipeline_result

    except Exception as e:
        _log_stage("PIPELINE", f"❌ Vessel segmentation exception: {e}")
        import traceback
        traceback.print_exc()
        pipeline_result["total_time"] = time.perf_counter() - pipeline_t0
        return pipeline_result

    # ── Stage 3: ROI Extraction ──────────────────────────────────────────
    try:
        roi_results = run_roi_extraction(
            vessel_output=vessel_output,
            output_dir=config.output_dir,
            save_roi=config.save_roi,
            save_seg=config.save_seg,
        )
        pipeline_result["roi_results"] = roi_results

    except Exception as e:
        _log_stage("PIPELINE", f"❌ ROI extraction exception: {e}")
        import traceback
        traceback.print_exc()
        pipeline_result["total_time"] = time.perf_counter() - pipeline_t0
        return pipeline_result

    # ── Final Summary ────────────────────────────────────────────────────
    total_time = time.perf_counter() - pipeline_t0
    pipeline_result["total_time"] = total_time
    pipeline_result["success"] = True

    _log_separator("PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  NIfTI path: {nifti_path}")
    if roi_results and roi_results.get("roi_shape"):
        print(f"  ROI shape:  {roi_results['roi_shape']}")
    if roi_results and roi_results.get("roi_spacing"):
        print(f"  ROI spacing: {roi_results['roi_spacing']}")
    if roi_results and roi_results.get("saved_files"):
        print(f"  Saved files:")
        for fp in roi_results["saved_files"]:
            print(f"    - {fp}")
    print()
    print("  Next steps:")
    print("    1. Use the ROI volume for downstream aneurysm classification")
    print("    2. Use the segmentation mask for vessel analysis")
    print("    3. Use transform_info for coordinate mapping back to DICOM space")
    print("=" * 70)

    return pipeline_result


# ============================================================================
# 9. CLI INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "RSNA 2025 End-to-End Pipeline: "
            "DICOM → NIfTI → Vessel Segmentation → ROI Extraction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline.py --patient-path ./data/patient_001/\n"
            "  python pipeline.py --patient-path ./data/patient_001/ --device cpu --save-roi\n"
            "  python pipeline.py --patient-path ./data/patient_001/ --output-dir ./results --save-roi --save-seg\n"
        ),
    )

    parser.add_argument(
        "--patient-path",
        type=str,
        required=True,
        help="Path to patient DICOM folder (single series)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default=DEFAULT_PRIMARY_MODEL,
        help="Path to primary dense nnU-Net model directory",
    )
    parser.add_argument(
        "--sparse-model",
        type=str,
        default=DEFAULT_SPARSE_MODEL,
        help="Path to sparse search nnU-Net model directory",
    )
    parser.add_argument(
        "--additional-model",
        type=str,
        default=DEFAULT_ADDITIONAL_DENSE_MODEL,
        help="Path to additional dense nnU-Net model directory",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help='Fold specification, comma-separated (default: "all")',
    )
    parser.add_argument(
        "--save-roi",
        action="store_true",
        help="Save ROI volume as .npy file",
    )
    parser.add_argument(
        "--save-seg",
        action="store_true",
        help="Save segmentation mask as .npy file",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse search (use full-volume dense inference)",
    )
    parser.add_argument(
        "--no-orientation-correction",
        action="store_true",
        help="Disable orientation correction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI usage."""
    args = parse_args()

    # Parse folds
    fold_tokens = [t.strip() for t in args.folds.split(",") if t.strip()]
    folds: List = []
    for token in fold_tokens:
        if token.lower() in ("all", "fold_all"):
            folds = ["all"]
            break
        try:
            folds.append(int(token))
        except ValueError:
            folds.append(token)
    if not folds:
        folds = ["all"]

    # Build config
    config = PipelineConfig(
        patient_path=args.patient_path,
        output_dir=args.output_dir,
        device=args.device,
        primary_model_path=args.primary_model,
        sparse_model_path=args.sparse_model,
        additional_dense_model_path=args.additional_model,
        folds=folds,
        use_sparse_search=not args.no_sparse,
        use_mirroring=False,
        enable_orientation_correction=not args.no_orientation_correction,
        verbose=not args.quiet,
        save_roi=args.save_roi,
        save_seg=args.save_seg,
    )

    # Run the full pipeline
    result = run_full_pipeline(
        patient_path=args.patient_path,
        config=config,
    )

    # Exit code
    if result["success"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
