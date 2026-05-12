"""多算法 RAW proxy-GT 生成工具.

读 probe_analysis/raw_proxyGT_data/raw_proxyGT_reference/<subject>/<hash>/raw_input_timeseries.csv,
对每种 rPPG 算法 (GREEN/CHROM/POS/LGI) 分别生成独立 proxy-GT .npz,
格式与 probe_analysis/GTPreprocessing/gt_reference.py 的 raw_video_rppg 源完全兼容.

生成物写到 probe_analysis/raw_proxyGT_data/proxy_gt/<subject>/<subject>_proxy_gt_<ALGO>.npz,
供后续 run_analysis 通过 gt_ref_proxy_by_algo dict 引用.

CLI 用法:
  # 单受试者:
  python probe_analysis/GTPreprocessing/generate_proxy_gt.py \
      --reference probe_analysis/raw_proxyGT_data/raw_proxyGT_reference/yjc/start_000300_frames_1600_bayer_GRBG_roi_e195f48be7/raw_input_timeseries.csv \
      --output-dir probe_analysis/raw_proxyGT_data/proxy_gt/yjc \
      --subject yjc --fps 30

  # 批量所有 subject:
  python probe_analysis/GTPreprocessing/generate_proxy_gt.py \
      --reference-root probe_analysis/raw_proxyGT_data/raw_proxyGT_reference \
      --output-root probe_analysis/raw_proxyGT_data/proxy_gt \
      --fps 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from multi_algo_bpm import (
    DEFAULT_STRIDE_SEC,
    DEFAULT_WINDOW_SIZE_SEC,
    MultiAlgoBpmError,
    SUPPORTED_ALGOS,
    compute_multi_algo_bpm,
    load_reference_rgb_from_csv,
)


PROXY_GT_FILENAME_TEMPLATE = "{subject}_proxy_gt_{algo}.npz"


def _read_reference_meta(reference_csv: str) -> dict:
    """若同目录存在 reference_meta.json, 读出; 否则返回空 dict."""
    meta_path = os.path.join(os.path.dirname(reference_csv), "reference_meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _pack_metadata(
    subject: str,
    algo: str,
    domain: str,
    reference_csv: str,
    fps: float,
    window_size_sec: float,
    stride_sec: float,
    n_windows: int,
    n_frames: int,
    ref_meta: dict,
) -> dict:
    """构造写入 .npz 的 metadata, 保留数据来源, 方便后续追溯/诊断."""
    return {
        "source_type": "raw_video_rppg",
        "subject": subject,
        "algo": algo,
        "domain": domain,
        "reference_csv": os.path.abspath(reference_csv),
        "reference_meta": ref_meta,
        "fps": float(fps),
        "window_size_sec": float(window_size_sec),
        "stride_sec": float(stride_sec),
        "n_windows": int(n_windows),
        "n_frames": int(n_frames),
        "bpm_method": "pyvhr_welch_0.65-4.0Hz_nfft2048",
        "hr_band_lo_hz": 0.65,
        "hr_band_hi_hz": 4.0,
    }


def generate_proxy_gt_from_reference(
    reference_csv: str,
    output_dir: str,
    subject: str,
    fps: float = 30.0,
    algos: Optional[List[str]] = None,
    window_size_sec: float = DEFAULT_WINDOW_SIZE_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    probe_time_mode: str = "absolute_frame_id",
    probe_time_offset_sec: float = 0.0,
    overwrite: bool = True,
) -> Dict[str, str]:
    """为一个 subject 生成多算法 proxy-GT .npz.

    Returns:
        dict {algo: output_npz_path}.
    """
    if not subject:
        raise MultiAlgoBpmError("subject is required.")
    os.makedirs(output_dir, exist_ok=True)

    signal, frame_ids, domain = load_reference_rgb_from_csv(reference_csv)
    ref_meta = _read_reference_meta(reference_csv)

    try:
        algo_results = compute_multi_algo_bpm(
            signal,
            frame_ids,
            fps,
            algos=algos,
            domain=domain,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            probe_time_mode=probe_time_mode,
            probe_time_offset_sec=probe_time_offset_sec,
            do_preprocess=True,
        )
    except MultiAlgoBpmError:
        raise
    except Exception as exc:
        raise MultiAlgoBpmError(
            f"compute_multi_algo_bpm failed for {subject}: {exc}"
        ) from exc

    output_paths: Dict[str, str] = {}
    n_frames = int(len(frame_ids))
    for algo, result in algo_results.items():
        bpm = np.asarray(result["bpm"], dtype=float)
        times = np.asarray(result["times"], dtype=float)
        if bpm.size == 0 or times.size == 0:
            print(
                f"[WARN] {subject}/{algo}: 窗口数为 0 (n_frames={n_frames}, "
                f"window={window_size_sec}s), 跳过写入."
            )
            continue
        # bpm 可能有 NaN (窗口 PSD 全零等极端情况), 按 gt_reference 要求过滤.
        finite_mask = np.isfinite(bpm) & np.isfinite(times)
        if finite_mask.sum() == 0:
            print(f"[WARN] {subject}/{algo}: 全部 BPM NaN, 跳过写入.")
            continue
        bpm = bpm[finite_mask]
        times = times[finite_mask]

        filename = PROXY_GT_FILENAME_TEMPLATE.format(subject=subject, algo=algo)
        out_path = os.path.join(output_dir, filename)
        if os.path.exists(out_path) and not overwrite:
            print(f"[SKIP] {out_path} 已存在且 overwrite=False, 跳过.")
            output_paths[algo] = out_path
            continue

        metadata = _pack_metadata(
            subject=subject,
            algo=algo,
            domain=domain,
            reference_csv=reference_csv,
            fps=fps,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            n_windows=int(bpm.size),
            n_frames=n_frames,
            ref_meta=ref_meta,
        )
        np.savez(
            out_path,
            bpm=bpm,
            times=times,
            metadata=np.array(metadata, dtype=object),
        )
        output_paths[algo] = out_path
        print(
            f"[OK] {subject}/{algo}: -> {out_path}  (n_win={bpm.size}, "
            f"bpm_mean={np.mean(bpm):.2f}, time={times[0]:.2f}-{times[-1]:.2f}s)"
        )
    return output_paths


def _find_reference_csv_for_subject(subject_dir: str) -> Optional[str]:
    """在 subject 目录下找 raw_input_timeseries.csv. 若有多个配置子目录, 取第一个."""
    candidates = sorted(glob.glob(os.path.join(subject_dir, "*", "raw_input_timeseries.csv")))
    if candidates:
        return candidates[0]
    direct = os.path.join(subject_dir, "raw_input_timeseries.csv")
    return direct if os.path.isfile(direct) else None


def batch_generate_for_root(
    reference_root: str,
    output_root: str,
    fps: float = 30.0,
    algos: Optional[List[str]] = None,
    window_size_sec: float = DEFAULT_WINDOW_SIZE_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    probe_time_mode: str = "absolute_frame_id",
    probe_time_offset_sec: float = 0.0,
    overwrite: bool = True,
) -> Dict[str, Dict[str, str]]:
    """遍历 reference_root 下所有 subject 目录批量生成."""
    if not os.path.isdir(reference_root):
        raise MultiAlgoBpmError(f"reference_root not found: {reference_root}")
    result: Dict[str, Dict[str, str]] = {}
    for subj_name in sorted(os.listdir(reference_root)):
        subj_dir = os.path.join(reference_root, subj_name)
        if not os.path.isdir(subj_dir):
            continue
        csv_path = _find_reference_csv_for_subject(subj_dir)
        if not csv_path:
            print(f"[SKIP] {subj_name}: 未找到 raw_input_timeseries.csv.")
            continue
        print(f"\n===== Subject: {subj_name}  CSV: {csv_path} =====")
        try:
            paths = generate_proxy_gt_from_reference(
                reference_csv=csv_path,
                output_dir=os.path.join(output_root, subj_name),
                subject=subj_name,
                fps=fps,
                algos=algos,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                probe_time_mode=probe_time_mode,
                probe_time_offset_sec=probe_time_offset_sec,
                overwrite=overwrite,
            )
            result[subj_name] = paths
        except MultiAlgoBpmError as exc:
            print(f"[ERROR] {subj_name}: {exc}")
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g_single = p.add_argument_group("single subject")
    g_single.add_argument("--reference", help="raw_input_timeseries.csv 路径")
    g_single.add_argument("--output-dir", help="单 subject 输出目录")
    g_single.add_argument("--subject", help="subject 名 (用于文件命名)")

    g_batch = p.add_argument_group("batch")
    g_batch.add_argument("--reference-root", help="raw_proxyGT_reference 根目录 (批处理)")
    g_batch.add_argument("--output-root", help="proxy_gt 根目录 (批处理)")

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--algos", nargs="*", default=None, choices=list(SUPPORTED_ALGOS),
                   help="要生成的算法列表, 默认全部")
    p.add_argument("--window-size-sec", type=float, default=DEFAULT_WINDOW_SIZE_SEC)
    p.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
    p.add_argument("--probe-time-mode", default="absolute_frame_id",
                   choices=["absolute_frame_id", "relative_csv"])
    p.add_argument("--probe-time-offset-sec", type=float, default=0.0)
    p.add_argument("--no-overwrite", action="store_true", help="已存在则跳过")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    algos = args.algos if args.algos else None
    overwrite = not args.no_overwrite

    if args.reference_root and args.output_root:
        batch_generate_for_root(
            reference_root=args.reference_root,
            output_root=args.output_root,
            fps=args.fps,
            algos=algos,
            window_size_sec=args.window_size_sec,
            stride_sec=args.stride_sec,
            probe_time_mode=args.probe_time_mode,
            probe_time_offset_sec=args.probe_time_offset_sec,
            overwrite=overwrite,
        )
        return

    if not (args.reference and args.output_dir and args.subject):
        raise SystemExit(
            "必须指定 (--reference + --output-dir + --subject) 或 (--reference-root + --output-root)."
        )
    generate_proxy_gt_from_reference(
        reference_csv=args.reference,
        output_dir=args.output_dir,
        subject=args.subject,
        fps=args.fps,
        algos=algos,
        window_size_sec=args.window_size_sec,
        stride_sec=args.stride_sec,
        probe_time_mode=args.probe_time_mode,
        probe_time_offset_sec=args.probe_time_offset_sec,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
