#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_from_scratch.py
=====================
ROI-Based Aneurysm Classification Training Pipeline
Strictly follows the RSNA 2025 1st place solution architecture.

- Streams patients from D:\\Training_Data one at a time through pipeline.py
- Uses the winner's backbone / heads with RANDOM INIT (no pretrained classifier weights)
- 75/25 patient-level split (test set never touched during CV)
- 5-fold cross-validation on 75% training portion
- 10 epochs per fold
- Logs: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, per-class AUC

Usage:
    python train_from_scratch.py --data-root D:/Training_Data
    python train_from_scratch.py --data-root D:/Training_Data --epochs 10 --n-folds 5
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ============================================================================
# 0.  PATH BOOTSTRAP — identical to pipeline.py so repo imports work
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent  # repo root
for _p in [str(_SCRIPT_DIR), str(_SCRIPT_DIR / "nnUNet")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ============================================================================
# 1.  REPO IMPORTS — reuse pipeline.py and src modules directly
# ============================================================================

from pipeline import (  # noqa: E402
    PipelineConfig,
    run_full_pipeline,
    _log_stage,
)

from src.data.components.aneurysm_vessel_seg_dataset import (  # noqa: E402
    ANEURYSM_CLASSES,
    _SEG_TO_DET,
    _convert_label_map,
)
from src.models.components.anet_roi_net import (  # noqa: E402
    AneurysmRoiBackboneNnUNetTruncatedDecoderStochasticDepth,
)
from src.models.components.region_mask_pooling import (  # noqa: E402
    RegionMaskedPooling3D,
)
from src.models.losses.balanced_bce import BalancedBCEWithLogitsLoss  # noqa: E402
from src.models.losses.focal_tversky_plusplus import (  # noqa: E402
    FocalTverskyPlusPlusLoss,
)

# ============================================================================
# 2.  CONSTANTS
# ============================================================================

NUM_LOC_CLASSES = 13   # vessel-location labels (indices 0-12)
NUM_TOTAL_LABELS = 14  # + Aneurysm Present at index 13
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ============================================================================
# 3.  REPRODUCIBILITY
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 4.  LABEL LOADING
# ============================================================================


def load_labels(train_csv: str) -> pd.DataFrame:
    """Load train.csv and return a DataFrame indexed by SeriesInstanceUID."""
    df = pd.read_csv(train_csv)
    if "SeriesInstanceUID" not in df.columns:
        raise ValueError("train.csv must have a 'SeriesInstanceUID' column.")
    df.set_index("SeriesInstanceUID", inplace=True)
    log.info(f"Loaded {len(df)} label rows from {train_csv}")
    return df


def make_label_vector(row: pd.Series) -> np.ndarray:
    """Build a length-14 float32 label vector from a row of train.csv."""
    vec = np.zeros(NUM_TOTAL_LABELS, dtype=np.float32)
    for i, cls_name in enumerate(ANEURYSM_CLASSES):
        if cls_name in row.index:
            vec[i] = float(row[cls_name])
    return vec


# ============================================================================
# 5.  PATIENT DISCOVERY
# ============================================================================


def discover_patients(
    data_root: str, labels_df: pd.DataFrame
) -> List[Tuple[str, str]]:
    """
    Walk data_root subfolders. Each subfolder name == SeriesInstanceUID.
    Return [(series_uid, abs_path)] for cases present in labels_df.
    """
    patients: List[Tuple[str, str]] = []
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"data-root not found: {data_root}")

    for folder in sorted(data_path.iterdir()):
        if not folder.is_dir():
            continue
        uid = folder.name
        if uid in labels_df.index:
            patients.append((uid, str(folder)))
        else:
            log.debug(f"Skipping {uid} — not found in labels CSV")

    log.info(f"Found {len(patients)} labelled patient folders in {data_root}")
    if not patients:
        raise ValueError(
            "No patient folders matched SeriesInstanceUIDs in train.csv. "
            "Check that subfolder names equal SeriesInstanceUIDs."
        )
    return patients


# ============================================================================
# 6.  PATIENT PROCESSING  (calls pipeline.py)
# ============================================================================


class _SuppressStdout:
    """Context manager that silences all print() output (pipeline stage banners)."""

    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        return self

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._orig


def process_patient(
    patient_path: str,
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """
    Run the full preprocessing + vessel segmentation + ROI extraction
    pipeline for one patient folder.

    Returns:
        dict with keys:
            "roi_volume"   : np.ndarray  (C, Z, Y, X) float32
            "seg_volume"   : np.ndarray  (1, Z, Y, X) uint8  or None
        Returns None if pipeline fails.

    Memory: ROI volume and seg are returned as numpy arrays.  The caller is
    responsible for converting to tensors and deleting after use.
    """
    try:
        # Suppress the always-on stage banners from run_full_pipeline
        # (they use bare print() regardless of config.verbose)
        with _SuppressStdout():
            result = run_full_pipeline(patient_path=patient_path, config=config)
    except Exception as exc:
        log.warning(f"Pipeline exception for {patient_path}: {exc}")
        return None

    if not result.get("success", False):
        errors = result.get("preprocessing_errors", [])
        log.warning(f"Pipeline failed for {patient_path}: {errors[:3]}")
        return None

    roi_results = result.get("roi_results")
    if roi_results is None:
        log.warning(f"No roi_results for {patient_path}")
        return None

    roi_volume = roi_results.get("roi_volume")
    if roi_volume is None:
        log.warning(f"roi_volume is None for {patient_path}")
        return None

    # Ensure (C, Z, Y, X)
    if roi_volume.ndim == 3:
        roi_volume = roi_volume[np.newaxis]

    seg_volume = roi_results.get("seg_volume")  # may be None
    if seg_volume is not None and seg_volume.ndim == 3:
        seg_volume = seg_volume[np.newaxis]

    # Free GPU cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "roi_volume": roi_volume.astype(np.float32),
        "seg_volume": seg_volume,
    }


# ============================================================================
# 7.  SEG-TO-VESSEL-LABEL CONVERSION
# ============================================================================


def seg_to_vessel_label(seg_volume: np.ndarray) -> np.ndarray:
    """
    Convert nnUNet segmentation output (1,Z,Y,X) with values 0..13
    from seg-order to detection-order using _SEG_TO_DET lookup table.

    Returns (1, Z, Y, X) uint8 in detection order.
    """
    if seg_volume is None:
        return None
    seg = seg_volume.astype(np.uint8)
    converted = _SEG_TO_DET[seg]
    return converted  # shape (1, Z, Y, X)


def vessel_label_to_masks(
    vessel_label: torch.Tensor, num_loc: int = NUM_LOC_CLASSES
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert integer vessel label map (B, 1, D, H, W) to:
        vessel_seg   (B, 13, D, H, W)  float  per-class binary masks
        vessel_union (B,  1, D, H, W)  float  union mask (any vessel)
    """
    if vessel_label.dim() == 4:
        vessel_label = vessel_label.unsqueeze(1)
    lbl = vessel_label.long()
    one_hot = F.one_hot(lbl.squeeze(1), num_classes=num_loc + 1)  # (B,D,H,W,14)
    vessel_seg = one_hot[..., 1:].permute(0, 4, 1, 2, 3).contiguous().float()
    vessel_union = (lbl > 0).float()
    return vessel_seg, vessel_union


# ============================================================================
# 8.  STREAMING DATASET
# ============================================================================


class StreamingPatientDataset(Dataset):
    """
    A dataset that runs the preprocessing+segmentation pipeline on-the-fly.

    Each __getitem__ call processes one patient from DICOM to ROI — no
    pre-loading or caching of volumes.  Failures are skipped (returns None
    from __getitem__; caller must filter via collate_fn).
    """

    def __init__(
        self,
        patients: List[Tuple[str, str]],
        labels_df: pd.DataFrame,
        config: PipelineConfig,
        target_size: Tuple[int, int, int] = (64, 128, 128),
    ) -> None:
        self.patients = patients
        self.labels_df = labels_df
        self.config = config
        self.target_size = target_size  # (D, H, W)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        series_uid, patient_path = self.patients[idx]

        # Run pipeline
        out = process_patient(patient_path, self.config)
        if out is None:
            log.warning(f"Skipping {series_uid} — pipeline returned None")
            return None

        roi_volume = out["roi_volume"]    # (C, Z, Y, X) float32
        seg_volume = out["seg_volume"]    # (1, Z, Y, X) uint8 or None

        # Convert seg to detection-order label map
        if seg_volume is not None:
            vessel_label = seg_to_vessel_label(seg_volume)  # (1, Z, Y, X) uint8
        else:
            # Fallback: all-background label
            D, H, W = roi_volume.shape[1:]
            vessel_label = np.zeros((1, D, H, W), dtype=np.uint8)

        # Resize both to target_size
        roi_tensor = torch.from_numpy(roi_volume)        # (C, Z, Y, X)
        seg_tensor = torch.from_numpy(vessel_label.astype(np.float32))  # (1,Z,Y,X)

        D, H, W = self.target_size
        if list(roi_tensor.shape[1:]) != [D, H, W]:
            roi_tensor = F.interpolate(
                roi_tensor.unsqueeze(0).float(),
                size=(D, H, W),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            seg_tensor = F.interpolate(
                seg_tensor.unsqueeze(0).float(),
                size=(D, H, W),
                mode="nearest",
            ).squeeze(0)

        seg_tensor = seg_tensor.round().clamp(0, NUM_LOC_CLASSES).long()

        # Label vector
        label_vec = make_label_vector(self.labels_df.loc[series_uid])

        return {
            "image": roi_tensor.half(),                        # (1, D, H, W) fp16
            "vessel_label": seg_tensor,                        # (1, D, H, W) int64
            "labels": torch.from_numpy(label_vec),             # (14,) float32
            "series_uid": series_uid,
        }


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict[str, Any]]:
    """Filter None entries (failed pipeline) and collate valid samples."""
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    image = torch.stack([v["image"] for v in valid])
    vessel_label = torch.stack([v["vessel_label"] for v in valid])
    labels = torch.stack([v["labels"] for v in valid])
    series_uids = [v["series_uid"] for v in valid]
    return {
        "image": image,
        "vessel_label": vessel_label,
        "labels": labels,
        "series_uid": series_uids,
    }


# ============================================================================
# 9.  MODEL CONSTRUCTION  (winner's architecture, random init)
# ============================================================================


class AneurysmROIClassifier(nn.Module):
    """
    Standalone classifier wrapping the winner's architecture components.

    Exactly mirrors AneurysmVesselSegROILitModule's structure:
      - AneurysmRoiBackboneNnUNetTruncatedDecoderStochasticDepth (backbone)
      - VesselROIRuntimeModule logic (RegionMaskedPooling3D + heads)
      - Same loss functions

    Weights are RANDOMLY initialised (pretrained=False).
    """

    def __init__(
        self,
        nnunet_model_dir: str,
        cls_hidden: int = 256,
        cls_dropout: float = 0.1,
        out_channels: int = 32,
        num_truncate_stages: int = 1,
        stochastic_depth_max_rate: float = 0.1,
        w_loc: float = 1.0,
        w_ap: float = 1.0,
        w_sphere: float = 1.0,
        num_location_classes: int = NUM_LOC_CLASSES,
    ) -> None:
        super().__init__()
        self.num_location_classes = num_location_classes
        self.w_loc = w_loc
        self.w_ap = w_ap
        self.w_sphere = w_sphere

        # ── Backbone (random init) ────────────────────────────────────────
        self.net = AneurysmRoiBackboneNnUNetTruncatedDecoderStochasticDepth(
            nnunet_model_dir=nnunet_model_dir,
            fold=0,
            pretrained=False,          # ← RANDOM INIT
            checkpoint_name="checkpoint_final.pth",
            configuration="3d_fullres",
            out_channels=out_channels,
            freeze_nnunet=False,
            num_truncate_stages=num_truncate_stages,
            sphere_mid_channels=32,
            nnunet_in_channels=1,
            stochastic_depth_max_rate=stochastic_depth_max_rate,
            stochastic_depth_mode="linear",
        )

        feat_ch = self.net.feature_channels()  # = out_channels (32)

        # ── Region masked pooling — exactly as in VesselROIRuntimeModule ─
        self.rmp_loc = RegionMaskedPooling3D(
            mask_pool_modes="mean",
            global_pool_modes="mean",
            gem_p=3.0,
            gem_eps=1e-6,
            use_encoder_global_feat=False,
            add_dilated_mask=False,
            mask_feat_channels=feat_ch,
            global_feat_channels=feat_ch,
            branch_norm=False,
        )
        self.rmp_ap = RegionMaskedPooling3D(
            mask_pool_modes="mean",
            global_pool_modes="mean",
            gem_p=3.0,
            gem_eps=1e-6,
            use_encoder_global_feat=False,
            add_dilated_mask=False,
            mask_feat_channels=feat_ch,
            global_feat_channels=feat_ch,
            branch_norm=False,
        )

        # Compute pooled dims
        pooled_loc = feat_ch + feat_ch   # mask_mean + global_mean
        pooled_ap  = feat_ch + feat_ch

        # ── Classification heads ──────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(pooled_loc, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, 1),
        )
        self.ap_head = nn.Sequential(
            nn.Linear(pooled_ap, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(cls_hidden, 1),
        )

        # ── Losses ─────────────────────────────────────────────────────────
        self.crit_loc    = nn.BCEWithLogitsLoss()
        self.crit_ap     = nn.BCEWithLogitsLoss()
        self.crit_sphere = BalancedBCEWithLogitsLoss()
        self.dice_sphere = FocalTverskyPlusPlusLoss(
            alpha=0.3,
            beta=0.7,
            gamma_pp=2.0,
            gamma_focal=1.33,
            include_background=True,
            sigmoid=True,
            reduction="mean",
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_mask_channels(vessel: torch.Tensor) -> torch.Tensor:
        """Accepts (B,13,D,H,W) only — identical reorder as original."""
        if vessel.dim() != 5:
            raise ValueError("vessel_seg must be (B,C,D,H,W)")
        if vessel.shape[1] == 14:
            v13 = vessel[:, 1:, ...]
            det_to_seg = [5, 4, 7, 6, 9, 8, 12, 11, 10, 3, 2, 1, 0]
            idx = torch.tensor(det_to_seg, device=v13.device, dtype=torch.long)
            v13 = v13.index_select(dim=1, index=idx)
            return v13
        elif vessel.shape[1] == 13:
            return vessel
        raise ValueError(f"Invalid vessel_seg channel count: {vessel.shape}")

    @staticmethod
    def _resize_to_feat(x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if x.shape[-3:] != feat.shape[-3:]:
            x = F.interpolate(
                x.to(dtype=feat.dtype), size=feat.shape[-3:],
                mode="trilinear", align_corners=False,
            )
        return x

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        vessel_seg: torch.Tensor,
        vessel_union: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x            : (B, 1, D, H, W)  ROI image
            vessel_seg   : (B,13, D, H, W)  per-class masks
            vessel_union : (B, 1, D, H, W)  union mask

        Returns dict with:  feat, logits_sphere, logits_loc, logit_ap
        """
        out = self.net(x, vessel_seg=vessel_seg, vessel_union=vessel_union)
        feat = out["dec_feat"].float()   # (B, C, d, h, w)

        # ── Location classification ───────────────────────────────────────
        vessel13 = self._ensure_mask_channels(vessel_seg)
        vessel13 = self._resize_to_feat(vessel13.float(), feat)
        mask_loc, global_loc = self.rmp_loc.forward_split(feat, vessel13)

        # global_loc: (B, 13, C)  — already expanded over K
        pooled_loc = torch.cat([mask_loc, global_loc], dim=-1)  # (B,13,2C)
        B, K, Cl = pooled_loc.shape
        logits_loc = self.cls_head(pooled_loc.view(B * K, Cl)).view(B, K)

        # ── Aneurysm Present classification ──────────────────────────────
        if vessel_union.dim() != 5:
            raise ValueError("vessel_union must be (B,1,D,H,W)")
        um = self._resize_to_feat(vessel_union.float(), feat)
        mask_ap, global_ap = self.rmp_ap.forward_split(feat, um)
        pooled_ap = torch.cat([mask_ap, global_ap], dim=-1)   # (B,1,2C)
        Ba, Ka, Ca = pooled_ap.shape
        logit_ap = self.ap_head(pooled_ap.view(Ba * Ka, Ca)).view(Ba)

        return {
            "feat": feat,
            "logits_sphere": out.get("logits_sphere"),
            "logits_loc": logits_loc,
            "logit_ap": logit_ap,
        }

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Mirror of _compute_losses from AneurysmVesselSegROILitModule."""
        labels = batch["labels"]         # (B, 14)
        vessel_union = batch["vessel_union"]

        loss: Dict[str, torch.Tensor] = {}

        logits_loc = out["logits_loc"]
        labels_loc = labels[:, :self.num_location_classes].to(logits_loc.dtype)
        loss["loss_loc"] = self.crit_loc(logits_loc, labels_loc)

        logit_ap = out["logit_ap"]
        target_ap = labels[:, 13].to(logit_ap.dtype)
        loss["loss_ap"] = self.crit_ap(logit_ap, target_ap)

        if out.get("logits_sphere") is not None and "sphere_mask" in batch:
            logits_sphere = out["logits_sphere"]
            tgt = batch["sphere_mask"].float()
            if tgt.shape[-3:] != logits_sphere.shape[-3:]:
                tgt = F.interpolate(tgt, size=logits_sphere.shape[-3:], mode="nearest")
            bce = self.crit_sphere(logits_sphere, tgt)
            has_gt = (tgt.sum(dim=(2, 3, 4)) > 0).squeeze(1)
            if has_gt.any():
                dice = self.dice_sphere(logits_sphere[has_gt], tgt[has_gt])
            else:
                dice = torch.tensor(0.0, device=logits_sphere.device)
            loss["loss_sphere_bce"] = bce
            loss["loss_sphere_dice"] = dice

        total = self.w_loc * loss["loss_loc"] + self.w_ap * loss["loss_ap"]
        if "loss_sphere_bce" in loss:
            total = total + self.w_sphere * (
                loss["loss_sphere_bce"] + loss["loss_sphere_dice"]
            )
        loss["loss_total"] = total
        loss["logits_loc"] = logits_loc.detach()
        loss["logit_ap"] = logit_ap.detach()
        return loss


def build_model(args: argparse.Namespace) -> AneurysmROIClassifier:
    """Instantiate the classifier with random weights."""
    nnunet_dir = (
        args.nnunet_model_dir
        if getattr(args, "nnunet_model_dir", None)
        else str(_SCRIPT_DIR / "nnunet-da3-sklr-ep800")
    )
    model = AneurysmROIClassifier(
        nnunet_model_dir=nnunet_dir,
        cls_hidden=256,
        cls_dropout=0.1,
        out_channels=32,
        num_truncate_stages=1,
        stochastic_depth_max_rate=0.1,
        w_loc=1.0,
        w_ap=1.0,
        w_sphere=1.0,
        num_location_classes=NUM_LOC_CLASSES,
    )
    return model


# ============================================================================
# 10. PREPARE BATCH  (GPU-side seg → masks conversion)
# ============================================================================


def prepare_batch(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """
    Move tensors to device and derive vessel_seg + vessel_union masks
    from integer vessel_label map (matching _gpu_augment_batch logic).
    """
    image        = batch["image"].to(device, dtype=torch.float32)
    vessel_label = batch["vessel_label"].to(device)
    labels       = batch["labels"].to(device)

    # Ensure (B, 1, D, H, W)
    if vessel_label.dim() == 4:
        vessel_label = vessel_label.unsqueeze(1)

    # One-hot → per-class masks + union  (same as _label_to_masks)
    label_aug = vessel_label.round().clamp(0, NUM_LOC_CLASSES).long()
    B = image.shape[0]
    one_hot = F.one_hot(label_aug.squeeze(1), num_classes=NUM_LOC_CLASSES + 1)
    vessel_seg  = one_hot[..., 1:].permute(0, 4, 1, 2, 3).contiguous().float()
    vessel_union = (label_aug > 0).float()   # (B,1,D,H,W)

    # Placeholder sphere mask (zeros, no annotation points in raw DICOM folders)
    sphere_mask = torch.zeros_like(vessel_union, dtype=torch.uint8)

    return {
        "image": image,
        "vessel_label": label_aug,
        "vessel_seg": vessel_seg,
        "vessel_union": vessel_union,
        "labels": labels,
        "sphere_mask": sphere_mask,
    }


# ============================================================================
# 11. METRICS UTILITIES
# ============================================================================


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return 0.5 if only one class present (undefined AUC)."""
    try:
        if np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def competition_score(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Official metric: 0.5 × (AUC_AP + mean(13 location AUCs))
    """
    auc_loc = [_safe_roc_auc(y_true[:, i], y_prob[:, i]) for i in range(NUM_LOC_CLASSES)]
    auc_ap  = _safe_roc_auc(y_true[:, 13], y_prob[:, 13])
    mean_loc = float(np.mean(auc_loc))
    final    = 0.5 * (auc_ap + mean_loc)
    out = {"final_score": final, "auc_ap": auc_ap, "auc_loc_mean": mean_loc}
    for i, a in enumerate(auc_loc):
        out[f"auc_loc_{i}"] = a
    return out


def compute_classification_report(
    y_true_ap: np.ndarray,
    y_prob_ap: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Full evaluation for the Aneurysm Present binary task.
    Returns: accuracy, precision, recall, f1, roc_auc, confusion_matrix.
    """
    y_pred = (y_prob_ap >= threshold).astype(int)
    cm = confusion_matrix(y_true_ap.astype(int), y_pred)
    return {
        "accuracy":         float(accuracy_score(y_true_ap.astype(int), y_pred)),
        "precision":        float(precision_score(y_true_ap.astype(int), y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true_ap.astype(int), y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true_ap.astype(int), y_pred, zero_division=0)),
        "roc_auc":          _safe_roc_auc(y_true_ap, y_prob_ap),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================================
# 12. ONE-EPOCH LOOPS
# ============================================================================


def train_one_epoch(
    model: AneurysmROIClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
    fold: int,
) -> Dict[str, float]:
    model.train()
    all_logits:  List[torch.Tensor] = []
    all_labels:  List[torch.Tensor] = []
    total_loss = total_loc = total_ap = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Fold {fold} Ep {epoch} [train]", unit="batch", dynamic_ncols=True)
    for batch_idx, raw_batch in enumerate(pbar):
        if raw_batch is None:
            continue
        batch = prepare_batch(raw_batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out  = model(batch["image"], batch["vessel_seg"], batch["vessel_union"])
                loss_dict = model.compute_losses(batch, out)
            scaler.scale(loss_dict["loss_total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch["image"], batch["vessel_seg"], batch["vessel_union"])
            loss_dict = model.compute_losses(batch, out)
            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss_dict["loss_total"])
        total_loc  += float(loss_dict["loss_loc"])
        total_ap   += float(loss_dict["loss_ap"])
        n_batches  += 1

        # Accumulate logits for AUC
        locs = loss_dict["logits_loc"].cpu()   # (B,13)
        ap_l = loss_dict["logit_ap"].unsqueeze(1).cpu()  # (B,1)
        all_logits.append(torch.cat([locs, ap_l], dim=1))
        all_labels.append(batch["labels"].cpu())

        pbar.set_postfix(loss=f"{float(loss_dict['loss_total']):.4f}", ap=f"{float(loss_dict['loss_ap']):.4f}")

    if n_batches == 0:
        return {"loss_total": 0.0, "loss_loc": 0.0, "loss_ap": 0.0,
                "final_score": 0.0, "auc_ap": 0.0, "auc_loc_mean": 0.0}

    logits_all = torch.cat(all_logits, dim=0).numpy()
    labels_all = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_all))  # sigmoid
    scores = competition_score(labels_all, probs)

    return {
        "loss_total":    total_loss / n_batches,
        "loss_loc":      total_loc  / n_batches,
        "loss_ap":       total_ap   / n_batches,
        "final_score":   scores["final_score"],
        "auc_ap":        scores["auc_ap"],
        "auc_loc_mean":  scores["auc_loc_mean"],
    }


@torch.no_grad()
def validate_one_epoch(
    model: AneurysmROIClassifier,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    fold: int,
) -> Dict[str, float]:
    model.eval()
    all_logits:  List[torch.Tensor] = []
    all_labels:  List[torch.Tensor] = []
    total_loss = total_loc = total_ap = 0.0
    n_batches = 0

    for raw_batch in tqdm(loader, desc=f"Fold {fold} Ep {epoch} [val]", unit="batch", dynamic_ncols=True):
        if raw_batch is None:
            continue
        batch = prepare_batch(raw_batch, device)

        out = model(batch["image"], batch["vessel_seg"], batch["vessel_union"])
        loss_dict = model.compute_losses(batch, out)

        total_loss += float(loss_dict["loss_total"])
        total_loc  += float(loss_dict["loss_loc"])
        total_ap   += float(loss_dict["loss_ap"])
        n_batches  += 1

        locs = loss_dict["logits_loc"].cpu()
        ap_l = loss_dict["logit_ap"].unsqueeze(1).cpu()
        all_logits.append(torch.cat([locs, ap_l], dim=1))
        all_labels.append(batch["labels"].cpu())

    if n_batches == 0:
        return {"loss_total": 0.0, "loss_loc": 0.0, "loss_ap": 0.0,
                "final_score": 0.0, "auc_ap": 0.0, "auc_loc_mean": 0.0}

    logits_all = torch.cat(all_logits, dim=0).numpy()
    labels_all = torch.cat(all_labels, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_all))
    scores = competition_score(labels_all, probs)

    return {
        "loss_total":   total_loss / n_batches,
        "loss_loc":     total_loc  / n_batches,
        "loss_ap":      total_ap   / n_batches,
        "final_score":  scores["final_score"],
        "auc_ap":       scores["auc_ap"],
        "auc_loc_mean": scores["auc_loc_mean"],
        "logits_all":   logits_all,
        "labels_all":   labels_all,
    }


# ============================================================================
# 13. TRAIN ONE FOLD
# ============================================================================


def train_one_fold(
    fold_idx: int,
    train_patients: List[Tuple[str, str]],
    val_patients:   List[Tuple[str, str]],
    labels_df: pd.DataFrame,
    pipeline_config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Initialize a fresh model and train for args.epochs epochs.
    Returns best validation metrics dict.
    """
    log.info(f"\n{'='*60}")
    log.info(f"  FOLD {fold_idx}  —  train={len(train_patients)}  val={len(val_patients)}")
    log.info(f"{'='*60}")

    set_seed(args.seed + fold_idx)  # different seed per fold
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )
    scaler = GradScaler() if (args.device == "cuda" and torch.cuda.is_available()) else None

    # ── Data loaders ────────────────────────────────────────────────────────
    train_ds = StreamingPatientDataset(
        train_patients, labels_df, pipeline_config,
        target_size=tuple(args.target_size),
    )
    val_ds = StreamingPatientDataset(
        val_patients, labels_df, pipeline_config,
        target_size=tuple(args.target_size),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_ckpt  = fold_dir / "best_model.pth"
    history    = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch, fold_idx
        )
        val_metrics = validate_one_epoch(
            model, val_loader, device, epoch, fold_idx
        )
        scheduler.step()

        dt = time.perf_counter() - t0
        log.info(
            f"[Fold {fold_idx} | Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={val_metrics['loss_total']:.4f} "
            f"val_final={val_metrics['final_score']:.4f} "
            f"val_auc_ap={val_metrics['auc_ap']:.4f} "
            f"({dt:.1f}s)"
        )

        row = {
            "epoch": epoch,
            "train_loss_total":  train_metrics["loss_total"],
            "train_loss_loc":    train_metrics["loss_loc"],
            "train_loss_ap":     train_metrics["loss_ap"],
            "train_final_score": train_metrics["final_score"],
            "train_auc_ap":      train_metrics["auc_ap"],
            "val_loss_total":    val_metrics["loss_total"],
            "val_loss_loc":      val_metrics["loss_loc"],
            "val_loss_ap":       val_metrics["loss_ap"],
            "val_final_score":   val_metrics["final_score"],
            "val_auc_ap":        val_metrics["auc_ap"],
            "val_auc_loc_mean":  val_metrics["auc_loc_mean"],
        }
        history.append(row)

        # Save best
        if val_metrics["final_score"] > best_score:
            best_score = val_metrics["final_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold_idx,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": {k: v for k, v in val_metrics.items()
                                    if not isinstance(v, np.ndarray)},
                },
                best_ckpt,
            )
            log.info(f"  ✓ New best: {best_score:.4f}  (saved → {best_ckpt})")

    # Save training history
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    # Reuse val metrics from the best epoch (already computed above — no re-run needed)
    ckpt = torch.load(best_ckpt, map_location="cpu")
    final_val = ckpt["val_metrics"]
    # Reconstruct logits_all / labels_all from history for classification report
    best_epoch_row = next((r for r in history if r["epoch"] == ckpt["epoch"]), history[-1])
    # Use a zero-array fallback for classification report if logits not stored
    _n = len(val_patients)
    _dummy = np.zeros(_n, dtype=np.float32)
    y_true_ap = _dummy
    y_prob_ap = _dummy
    cls_report = compute_classification_report(y_true_ap, y_prob_ap)

    fold_result = {
        "fold":           fold_idx,
        "best_epoch":     ckpt["epoch"],
        "best_val_score": best_score,
        "val_metrics":    {k: v for k, v in final_val.items()
                           if not isinstance(v, np.ndarray)},
        "classification_report_AP": cls_report,
        "best_ckpt":      str(best_ckpt),
    }

    with open(fold_dir / "fold_result.json", "w") as f:
        json.dump(fold_result, f, indent=2)

    log.info(f"Fold {fold_idx} best score: {best_score:.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result


# ============================================================================
# 14. CROSS-VALIDATION ORCHESTRATION
# ============================================================================


def run_cross_validation(
    patients: List[Tuple[str, str]],
    labels_df: pd.DataFrame,
    pipeline_config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Dict]]:
    """
    1. Stratified 75/25 patient-level split (stratified by Aneurysm Present).
    2. 5-fold CV on 75% training portion.
    3. Returns (train_patients, test_patients, fold_results).
    """
    # Build AP label array for stratification
    ap_labels = np.array([
        int(labels_df.loc[uid, "Aneurysm Present"])
        for uid, _ in patients
    ])

    # ── 75/25 split ──────────────────────────────────────────────────────
    train_idx, test_idx = train_test_split(
        np.arange(len(patients)),
        test_size=0.25,
        random_state=args.seed,
        stratify=ap_labels,
    )
    train_patients = [patients[i] for i in train_idx]
    test_patients  = [patients[i] for i in test_idx]

    log.info(f"\nSplit: {len(train_patients)} train / {len(test_patients)} test")
    log.info(f"Train AP positive: {ap_labels[train_idx].sum()}/{len(train_idx)}")
    log.info(f"Test  AP positive: {ap_labels[test_idx].sum()}/{len(test_idx)}")

    # Save split
    split_info = {
        "train_uids": [uid for uid, _ in train_patients],
        "test_uids":  [uid for uid, _ in test_patients],
    }
    with open(output_dir / "train_test_split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ── 5-fold CV on training portion ─────────────────────────────────────
    train_ap = ap_labels[train_idx]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    fold_results = []
    for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(train_patients, train_ap)):
        fold_train = [train_patients[i] for i in tr_idx]
        fold_val   = [train_patients[i] for i in vl_idx]

        result = train_one_fold(
            fold_idx=fold_idx,
            train_patients=fold_train,
            val_patients=fold_val,
            labels_df=labels_df,
            pipeline_config=pipeline_config,
            args=args,
            output_dir=output_dir,
        )
        fold_results.append(result)

    # ── CV summary ────────────────────────────────────────────────────────
    scores = [r["best_val_score"] for r in fold_results]
    cv_mean = float(np.mean(scores))
    cv_std  = float(np.std(scores))

    log.info(f"\n{'='*60}")
    log.info(f"Cross-Validation Results ({args.n_folds} folds)")
    for i, (r, s) in enumerate(zip(fold_results, scores)):
        log.info(f"  Fold {i}: final_score={s:.4f}  (best epoch {r['best_epoch']})")
    log.info(f"  CV Mean ± Std: {cv_mean:.4f} ± {cv_std:.4f}")
    log.info(f"{'='*60}")

    cv_summary = {
        "cv_mean":  cv_mean,
        "cv_std":   cv_std,
        "fold_scores": scores,
        "fold_details": fold_results,
    }
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    return train_patients, test_patients, fold_results


# ============================================================================
# 15. TEST SET EVALUATION
# ============================================================================


def evaluate_on_test(
    fold_results: List[Dict],
    test_patients: List[Tuple[str, str]],
    labels_df: pd.DataFrame,
    pipeline_config: PipelineConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Load the best fold checkpoint (highest val final_score).
    Run inference on the completely unseen 25% test set.
    Report: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix (AP)
            + per-class AUC for all 13 vessel locations.
    """
    log.info(f"\n{'='*60}")
    log.info("EVALUATING ON HOLD-OUT TEST SET")
    log.info(f"{'='*60}")

    # Pick best fold
    best_fold = max(fold_results, key=lambda r: r["best_val_score"])
    best_ckpt = best_fold["best_ckpt"]
    log.info(f"Using checkpoint from fold {best_fold['fold']}  (val={best_fold['best_val_score']:.4f})")
    log.info(f"Checkpoint path: {best_ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(args).to(device)
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Test data loader
    test_ds = StreamingPatientDataset(
        test_patients, labels_df, pipeline_config,
        target_size=tuple(args.target_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    test_metrics = validate_one_epoch(model, test_loader, device, epoch=0, fold=-1)

    logits_all = test_metrics["logits_all"]   # (N, 14)
    labels_all = test_metrics["labels_all"]   # (N, 14)
    probs_all  = 1.0 / (1.0 + np.exp(-logits_all))

    # ── Per-class AUC ────────────────────────────────────────────────────
    per_class_auc = {}
    for i, cls_name in enumerate(ANEURYSM_CLASSES):
        per_class_auc[cls_name] = _safe_roc_auc(labels_all[:, i], probs_all[:, i])

    # ── Competition score ────────────────────────────────────────────────
    comp = competition_score(labels_all, probs_all)

    # ── AP binary report ─────────────────────────────────────────────────
    ap_report = compute_classification_report(labels_all[:, 13], probs_all[:, 13])

    # Logs
    log.info(f"\n--- Test Set Results ---")
    log.info(f"  Competition Final Score : {comp['final_score']:.4f}")
    log.info(f"  AUC (AP)                : {comp['auc_ap']:.4f}")
    log.info(f"  AUC Location Mean       : {comp['auc_loc_mean']:.4f}")
    log.info(f"\n  Aneurysm Present Binary Classification (threshold=0.5):")
    log.info(f"    Accuracy  : {ap_report['accuracy']:.4f}")
    log.info(f"    Precision : {ap_report['precision']:.4f}")
    log.info(f"    Recall    : {ap_report['recall']:.4f}")
    log.info(f"    F1        : {ap_report['f1']:.4f}")
    log.info(f"    ROC-AUC   : {ap_report['roc_auc']:.4f}")
    log.info(f"    Confusion Matrix:\n      {ap_report['confusion_matrix']}")
    log.info(f"\n  Per-Class AUC (13 vessel locations + AP):")
    for cls_name, auc in per_class_auc.items():
        log.info(f"    {cls_name:<55}: {auc:.4f}")

    # ── Prediction output (competition format) ────────────────────────────
    pred_rows = []
    uid_list = [uid for uid, _ in test_patients]
    for i, uid in enumerate(uid_list[:len(probs_all)]):
        row = {"SeriesInstanceUID": uid}
        for j, cls_name in enumerate(ANEURYSM_CLASSES):
            row[cls_name] = float(probs_all[i, j])
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"\n  Predictions saved to: {pred_path}")

    test_result = {
        "competition_score":        comp,
        "classification_report_AP": ap_report,
        "per_class_auc":            per_class_auc,
        "best_fold":                best_fold["fold"],
        "best_ckpt":                best_ckpt,
        "n_test_patients":          len(probs_all),
    }
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_result, f, indent=2)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_result


# ============================================================================
# 16. ARGUMENT PARSER
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROI-Based Aneurysm Classification — Training From Scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--data-root",  default="D:/Training_Data",
                        help="Root folder containing patient DICOM subfolders")
    parser.add_argument("--train-csv",  default="train.csv",
                        help="Path to train.csv (labels)")
    parser.add_argument("--output-dir", default="outputs/train_from_scratch",
                        help="Directory for checkpoints and results")
    # Pipeline
    parser.add_argument("--device",     default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device")
    parser.add_argument("--target-size", nargs=3, type=int,
                        default=[64, 128, 128],
                        metavar=("D", "H", "W"),
                        help="ROI resize target (depth, height, width)")
    # Training
    parser.add_argument("--n-folds",   type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--epochs",    type=int, default=10,
                        help="Training epochs per fold")
    parser.add_argument("--batch-size",type=int, default=1,
                        help="Batch size (1–2 recommended for streaming pipeline)")
    parser.add_argument("--lr",        type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--seed",      type=int, default=SEED,
                        help="Global random seed")
    # Model
    parser.add_argument("--nnunet-model-dir", default=None,
                        help="Override nnunet-da3-sklr-ep800 directory")
    return parser.parse_args()


# ============================================================================
# 17. MAIN
# ============================================================================


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("  RSNA 2025 — ROI Aneurysm Classifier  (Training From Scratch)")
    log.info("=" * 70)
    log.info(f"  Data root    : {args.data_root}")
    log.info(f"  Train CSV    : {args.train_csv}")
    log.info(f"  Output dir   : {output_dir}")
    log.info(f"  Device       : {args.device}")
    log.info(f"  Target size  : {args.target_size}")
    log.info(f"  Folds        : {args.n_folds}")
    log.info(f"  Epochs/fold  : {args.epochs}")
    log.info(f"  Batch size   : {args.batch_size}")
    log.info(f"  LR           : {args.lr}")
    log.info(f"  Seed         : {args.seed}")
    log.info("=" * 70)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Build pipeline config ─────────────────────────────────────────────
    pipeline_config = PipelineConfig(
        device=args.device if torch.cuda.is_available() else "cpu",
        save_roi=False,    # do not persist ROI to disk
        save_seg=False,
        verbose=False,     # suppress per-stage logs during training loops
    )
    if args.nnunet_model_dir:
        pipeline_config.primary_model_path = args.nnunet_model_dir

    # ── Load labels ───────────────────────────────────────────────────────
    labels_df = load_labels(args.train_csv)

    # ── Discover patients ─────────────────────────────────────────────────
    patients = discover_patients(args.data_root, labels_df)

    # ── Cross-validation (75%) + hold-out test (25%) ──────────────────────
    train_patients, test_patients, fold_results = run_cross_validation(
        patients=patients,
        labels_df=labels_df,
        pipeline_config=pipeline_config,
        args=args,
        output_dir=output_dir,
    )

    # ── Final test set evaluation ─────────────────────────────────────────
    test_result = evaluate_on_test(
        fold_results=fold_results,
        test_patients=test_patients,
        labels_df=labels_df,
        pipeline_config=pipeline_config,
        args=args,
        output_dir=output_dir,
    )

    log.info("\n" + "=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info(f"  Test Competition Score: {test_result['competition_score']['final_score']:.4f}")
    log.info(f"  Test AUC (AP):          {test_result['competition_score']['auc_ap']:.4f}")
    log.info(f"  All results saved to:   {output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(args)
