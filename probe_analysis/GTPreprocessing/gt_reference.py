"""Ground-truth BPM reference interface for probe analysis.

The probe analysis code consumes GT through this module instead of depending on
a concrete data source. The current implementation supports contact PPG GT from
Data_for_pyVHR/gt_data. A file-based BPM reference path is kept for future RAW
video rPPG-derived GT, as long as that source exports bpm/times arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional, Tuple

import numpy as np


DEFAULT_GT_ROOT = "Data_for_pyVHR/gt_data"
CONTACT_PPG_SOURCE = "contact_ppg"
RAW_VIDEO_RPPG_SOURCE = "raw_video_rppg"


class GTReferenceError(ValueError):
    """Raised when a GT reference cannot be loaded or aligned."""


def _metadata_to_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif len(value) == 1:
            value = value[0]
    return value if isinstance(value, dict) else {}


@dataclass
class BPMGTReference:
    """BPM/time GT reference used by probe analysis."""

    subject: str
    source_type: str
    bpm: np.ndarray
    times: np.ndarray
    data_path: str
    source_dir: str = ""
    metadata: dict = field(default_factory=dict)
    time_origin: str = "relative_seconds"

    def __post_init__(self) -> None:
        self.bpm = np.asarray(self.bpm, dtype=float).reshape(-1)
        self.times = np.asarray(self.times, dtype=float).reshape(-1)
        if self.bpm.size == 0 or self.times.size == 0:
            raise GTReferenceError("GT bpm/times arrays must be non-empty.")
        if self.bpm.size != self.times.size:
            raise GTReferenceError(
                f"GT bpm/times length mismatch: {self.bpm.size} vs {self.times.size}."
            )
        valid = np.isfinite(self.bpm) & np.isfinite(self.times)
        if not np.any(valid):
            raise GTReferenceError("GT bpm/times arrays contain no finite samples.")
        self.bpm = self.bpm[valid]
        self.times = self.times[valid]
        order = np.argsort(self.times)
        self.times = self.times[order]
        self.bpm = self.bpm[order]

    @classmethod
    def from_npz(
        cls,
        path: str,
        subject: str,
        source_type: str,
        source_dir: str = "",
    ) -> "BPMGTReference":
        if not os.path.exists(path):
            raise GTReferenceError(f"GT file not found: {path}")
        data = np.load(path, allow_pickle=True)
        required = {"bpm", "times"}
        missing = required.difference(data.files)
        if missing:
            raise GTReferenceError(f"GT file missing keys {sorted(missing)}: {path}")
        metadata = _metadata_to_dict(data["metadata"]) if "metadata" in data.files else {}
        return cls(
            subject=subject,
            source_type=source_type,
            bpm=data["bpm"],
            times=data["times"],
            data_path=path,
            source_dir=source_dir or os.path.dirname(path),
            metadata=metadata,
        )

    @classmethod
    def from_contact_ppg(
        cls,
        subject: str,
        gt_root: str = DEFAULT_GT_ROOT,
    ) -> "BPMGTReference":
        subject = _normalize_subject(subject)
        gt_dir = os.path.join(gt_root, f"gt_{subject}")
        npz_path = os.path.join(gt_dir, "bpms_times_GT.npz")
        return cls.from_npz(
            npz_path,
            subject=subject,
            source_type=CONTACT_PPG_SOURCE,
            source_dir=gt_dir,
        )

    @classmethod
    def from_raw_video_rppg_file(
        cls,
        path: str,
        subject: str = "",
    ) -> "BPMGTReference":
        if not path:
            raise GTReferenceError(
                "raw_video_rppg GT requires a file path exporting bpm/times arrays."
            )
        return cls.from_npz(
            path,
            subject=_normalize_subject(subject) if subject else "",
            source_type=RAW_VIDEO_RPPG_SOURCE,
            source_dir=os.path.dirname(path),
        )

    def available_time_range(self) -> Tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    def sampling_rate_hint(self) -> Optional[float]:
        rate = self.metadata.get("sampling_rate")
        try:
            return float(rate)
        except (TypeError, ValueError):
            return None

    def bpm_at(
        self,
        query_times: np.ndarray,
        method: str = "nearest",
        max_time_diff_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Return GT BPM values at query times.

        method="nearest" matches pyVHR error metrics, which compare each
        estimated BPM timestamp to the nearest GT timestamp.

        method="pyvhr_index" matches pyVHR get_SNR(), where the GT BPM is
        selected as reference_hrs[int(timesES[idx])].
        """
        qt = np.asarray(query_times, dtype=float).reshape(-1)
        out = np.full(qt.shape, np.nan, dtype=float)
        finite = np.isfinite(qt)
        if not np.any(finite):
            return out

        q = qt[finite]
        if method == "pyvhr_index":
            idx = q.astype(int)
            ok = (idx >= 0) & (idx < self.bpm.size)
            vals = np.full(q.shape, np.nan, dtype=float)
            vals[ok] = self.bpm[idx[ok]]
            out[finite] = vals
            return out

        if method == "interpolate":
            in_range = (q >= self.times[0]) & (q <= self.times[-1])
            vals = np.full(q.shape, np.nan, dtype=float)
            vals[in_range] = np.interp(q[in_range], self.times, self.bpm)
            out[finite] = vals
            return out

        if method != "nearest":
            raise GTReferenceError(f"Unsupported GT lookup method: {method}")

        idx = np.searchsorted(self.times, q, side="left")
        idx = np.clip(idx, 0, self.times.size - 1)
        prev_idx = np.clip(idx - 1, 0, self.times.size - 1)
        use_prev = np.abs(self.times[prev_idx] - q) <= np.abs(self.times[idx] - q)
        nearest_idx = np.where(use_prev, prev_idx, idx)
        diff = np.abs(self.times[nearest_idx] - q)
        vals = self.bpm[nearest_idx].astype(float)
        if max_time_diff_sec is not None:
            vals[diff > max_time_diff_sec] = np.nan
        out[finite] = vals
        return out

    def describe(self) -> dict:
        start, end = self.available_time_range()
        return {
            "GT_Source": self.source_type,
            "GT_Subject": self.subject,
            "GT_Data_Path": self.data_path,
            "GT_Time_Start_s": start,
            "GT_Time_End_s": end,
            "GT_BPM_Count": int(self.bpm.size),
            "GT_BPM_Mean_All": float(np.mean(self.bpm)),
            "GT_BPM_Min_All": float(np.min(self.bpm)),
            "GT_BPM_Max_All": float(np.max(self.bpm)),
        }


def _normalize_subject(subject: str) -> str:
    if not subject:
        raise GTReferenceError("GT subject is required.")
    subject = str(subject).strip()
    return subject[3:] if subject.startswith("gt_") else subject


def load_gt_reference(
    source_type: str = CONTACT_PPG_SOURCE,
    subject: str = "",
    gt_root: str = DEFAULT_GT_ROOT,
    path: str = "",
) -> BPMGTReference:
    """Factory used by CLI/batch code to load a GT reference."""
    if source_type in ("none", "", None):
        raise GTReferenceError("GT source_type cannot be none when loading GT.")
    if source_type == CONTACT_PPG_SOURCE:
        return BPMGTReference.from_contact_ppg(subject=subject, gt_root=gt_root)
    if source_type == RAW_VIDEO_RPPG_SOURCE:
        return BPMGTReference.from_raw_video_rppg_file(path=path, subject=subject)
    raise GTReferenceError(f"Unsupported GT source_type: {source_type}")
