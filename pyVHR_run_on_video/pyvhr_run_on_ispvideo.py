import pyVHR as vhr
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
import numpy as np


# -- PARAMETER SETTING and GT
fps = 30

# --------------------------------------------------------------------------
# 1. 分析函数 (analyze_video) 
# --------------------------------------------------------------------------

def analyze_video(video_path, pipe, video_id=1, wsize=8, 
                  roi_method='convexhull', method='cpu_CHROM', roi_approach='holistic', estimate='bpm_est'):
    """ 
    分析单个视频的心率特征，并打印结果。
    参数:
        video_path : str   视频文件路径
        pipe       : Pipeline 对象
        video_id   : int   视频编号（打印时使用）
        wsize      : int   时间窗口大小
        roi_method : str   确定roi使用的算法
        method     : str   使用的rPPG算法（默认cpu_CHROM）
        estimate   ：str   BPM final estimate, if patches choose 'medians' or 'clustering'

    """
    # --- 运行 pyVHR 管线 ---
    bvps, timesES, bpmES = pipe.run_on_video(
        videoFileName=video_path,
        winsize=wsize, 
        roi_method=roi_method,
        roi_approach=roi_approach,
        method=method,
        estimate=estimate,
        patch_size=40, 
        RGB_LOW_HIGH_TH=(75,230),
        Skin_LOW_HIGH_TH=(75,230),
        pre_filt=True,
        post_filt=True,
        cuda=True, 
        verb=True
    )

    # 检查是否有有效输出
    if bpmES is None or timesES is None or len(bpmES) == 0:
        print(f"警告: 视频 {video_id}未能成功估算出BPM值。")
        return None, None, None
    
    # 打印每个时间戳的结果
    print(f"视频{video_id}估算出的BPM值:", bpmES)
    print(f"视频{video_id}对应的时间点 (秒):", timesES)

    # --- 转换为列表 ---
    bpm_values = [b.item() for b in bpmES]

    # --- 计算统计指标 ---
    mean_bpm = np.mean(bpm_values)
    std_bpm = np.std(bpm_values)
    min_bpm = np.min(bpm_values)
    max_bpm = np.max(bpm_values)

    # --- 打印结果 ---
    print(f"\n视频{video_id}的心率参数：")
    print(f"\t平均心率: {mean_bpm:.2f} BPM")
    print(f"\t标准差: {std_bpm:.2f} BPM")
    print(f"\t心率范围: 从 {min_bpm:.2f} BPM 到 {max_bpm:.2f} BPM")

    return bvps, timesES, bpm_values


# --------------------------------------------------------------------------
# 2. 主程序部分 - 在这里定义要处理的视频列表并调用分析函数
# --------------------------------------------------------------------------

# --- 在这里填入视频文件路径列表 ---
video_files = [
    'output_video_16bit_lossless__raw16_2raw.mkv'
]

# 用于存储所有视频的分析结果
all_results = []

# 实例化 Pipeline 对象
pipe = Pipeline() 

# 循环处理每个视频文件
for i, video_path in enumerate(video_files):
    # 调用分析函数，传入视频路径和ID
    # 注意：这里假设 pipe 对象已经创建
    _, times, bpms = analyze_video(video_path, pipe, video_id=i+1, method='cpu_LGI')
    
    # 如果结果有效，则存储起来
    if times is not None and bpms is not None:
        all_results.append({
            'id': f'Video {i+1}',
            'times': times,
            'bpms': bpms
        })


# --------------------------------------------------------------------------
# 3. 统一绘图部分 - 将所有结果绘制在同一张图上
# --------------------------------------------------------------------------
try:
    # 检查是否至少有一个有效结果用于绘图
    if all_results:
        plt.figure(figsize=(14, 7))

        # 循环遍历存储的每个视频的结果，并绘制曲线
        for result in all_results:
            plt.plot(result['times'], result['bpms'], label=f'Heart rate for {result["id"]}', marker='o', linestyle='-')

        # 设置图表属性
        plt.xlabel("Time (seconds)")
        plt.ylabel("BPM (Beats Per Minute)")
        plt.title("Comparison of Heart Rate Estimations")
        plt.legend()  # 显示图例，区分不同曲线
        plt.grid(True)
        
        # --- 将图像保存到文件 ---
        output_filename = 'HR_plot_comparison_of_8bitVS16bit_raw16_2raw.png'
        plt.savefig(output_filename)
        print(f"\n图像已成功保存到: {output_filename}")
        plt.show() # 如果您想在运行时直接看到图像，可以取消此行注释

    else:
        print("\n未能成功估算出任何视频的BPM值，无法绘制图像。")

except Exception as e:
    print(f"\n处理或绘图时出错: {e}")
    print("请检查视频路径是否正确，以及视频中是否能检测到人脸。")



# try:
#     # 打印或绘制估算结果
#     if bpmES is not None and timesES is not None:

#         # 为了方便绘图和计算，将 bpmES 转换为简单列表
#         bpm_values = [b.item() for b in bpmES_1]

#         # 使用 matplotlib 绘制结果图
#         plt.figure(figsize=(12, 6))
#         plt.plot(timesES_1, bpm_values, label='Estimated heart rate (BPM)', marker='o')
#         plt.xlabel("time (sencond)")
#         plt.ylabel("BPM")
#         plt.title("Heart rate estimation result")
#         plt.legend()
#         plt.grid(True)
        
#         # --- 将图像保存到文件，文件名可以自定义 ---
#         output_filename = 'HR_plot_8bit_lossless__raw16_2raw.png'
#         plt.savefig(output_filename)
#         print(f"图像已成功保存到: {output_filename}")

#     else:
#         print("未能成功估算出BPM值，无法绘制图像。")

# except Exception as e:
#     print(f"处理或绘图时出错: {e}")
#     print("请检查视频路径是否正确，以及视频中是否能检测到人脸。")

# ERRORS
# RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
# printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)  # 文本报告，在终端打印rPPG性能评估指标
# displayErrors(bpmES, bpmGT, timesES, timesGT) # 绘制估测值和真实值的曲线