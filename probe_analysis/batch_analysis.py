"""
ISP 探针批量对比分析
修改下方 ROOT_DIR / SUBJECTS / FPS 后直接运行即可。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from probe_ana_cmp import run_analysis

# ============================================================
# 用户配置区（按需修改）
# ============================================================

# 单受试者模式：直接指定探针根目录
ROOT_DIR = "isp_debug_probes/baseenv_baseISP/lzz"

# 多受试者模式：填入受试者名列表，留空则只处理 ROOT_DIR
# 格式：["hyl", "wyx", "yjc"]，根目录会自动替换为 .../baseenv_baseISP/<subject>
# 注意ROOT_DIR最后需要取到具体人名，因为后续会截取上一级目录
SUBJECTS = ["zbw","lxr"]

FPS = 30.0

# 输出根目录（每个配对结果保存在其子目录中）
OUTPUT_BASE = "probe_analysis/probe_analysis_result/baseenv_baseISP_test"

# 固定配对列表：(探针1目录名, 探针2目录名, 输出子目录名)
PROBE_PAIRS = [
    # RAW 域内部
    ("Input-BlackLevel",               "BlackLevel-DefectPixel",            "01_BLC"),
    ("BlackLevel-DefectPixel",       "DefectPixel-WhiteBalance",          "02_DPC"),
    ("DefectPixel-WhiteBalance",     "WhiteBalance-Demosaic",             "03_WB"),
    # 跨域：RAW → RGB
    ("WhiteBalance-Demosaic",        "Demosaic-CCM",                      "04_Demosaic"),
    # RGB 域内部
    ("Demosaic-CCM",                 "CCM-Gamma",                         "05_CCM"),
    ("CCM-Gamma",                    "Gamma-ColorSpace",                  "06_Gamma"),
    # 跨域：RGB → YUV
    ("Gamma-ColorSpace",             "ColorSpace-ContrastSaturation",     "07_ColorSpace"),
    # YUV 域内部
    ("ColorSpace-ContrastSaturation","ContrastSaturation-YUVtoRGB",       "08_ContrastSat"),
    # 跨域：YUV → RGB
    ("ContrastSaturation-YUVtoRGB",  "YUVtoRGB-Output",                   "09_YUV2RGB"),
]

# ============================================================

def batch_run(root_dir: str, output_base: str, subject: str = ""):
    label = subject if subject else os.path.basename(root_dir)
    print(f"\n{'='*50}")
    print(f"受试者: {label}  根目录: {root_dir}")
    print(f"{'='*50}")
    ok, skip, fail = 0, 0, 0
    for probe1, probe2, out_name in PROBE_PAIRS:
        csv1 = os.path.join(root_dir, probe1, f"{probe1}_timeseries.csv")
        csv2 = os.path.join(root_dir, probe2, f"{probe2}_timeseries.csv")
        if not os.path.exists(csv1) or not os.path.exists(csv2):
            print(f"  [跳过] {probe1} 或 {probe2} CSV 不存在")
            skip += 1
            continue
        out_dir = os.path.join(output_base, label, out_name)
        print(f"\n  [{out_name}] {probe1} vs {probe2}")
        try:
            run_analysis(csv1, csv2, FPS, out_dir)
            ok += 1
        except Exception as e:
            print(f"  [错误] {e}")
            fail += 1
    print(f"\n  完成: {ok} 成功 / {skip} 跳过 / {fail} 失败")


if __name__ == "__main__":
    if SUBJECTS:
        base_parent = os.path.dirname(ROOT_DIR)
        for subj in SUBJECTS:
            root = os.path.join(base_parent, subj)
            batch_run(root, OUTPUT_BASE, subj)
    else:
        batch_run(ROOT_DIR, OUTPUT_BASE)
