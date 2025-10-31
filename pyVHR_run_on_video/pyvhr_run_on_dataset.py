import pyVHR as vhr
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
import numpy as np

# -- SET DATASET，使用标准数据集时才需要

dataset_name = 'PURE'                      # the name of the python class handling it 
video_DIR = '/var/datasets/VHR1/PURE/'  # dir containing videos
BVP_DIR = '/var/datasets/VHR1/PURE/'    # dir containing BVPs GT

dataset = vhr.datasets.datasetFactory(dataset_name, videodataDIR=video_DIR, BVPdataDIR=BVP_DIR)
allvideo = dataset.videoFilenames

# print the list of video names with the progressive index (idx)
for v in range(len(allvideo)):
  print(v, allvideo[v])


# -- PARAMETER SETTING and GT
wsize = 6                  # seconds of video processed (with overlapping) for each estimate
video_idx = 1     # index of the video to be processed
fname = dataset.getSigFilename(video_idx)
sigGT = dataset.readSigfile(fname)   #获取ground_truth文件中的GT数据，其中包含了BVP、HR与对应时间戳
test_bvp = sigGT.data
bpmGT, timesGT = sigGT.getBPM(wsize)  # bvpGT——>bpmGT
fps = 30

roi_approach = 'holistic'   # 'holistic' or 'patches'
bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
method = 'cpu_CHROM'       # one of the methods implemented in pyVHR
my_video_path = "pyVHR_project1/UBFC2/subject1/vid.avi"


# run
pipe = Pipeline()          # object to execute the pipeline
bvps, timesES, bpmES = pipe.run_on_video(videoFileName=my_video_path,
                                        winsize=wsize, 
                                        roi_method='convexhull',
                                        roi_approach=roi_approach,
                                        method=method,
                                        estimate=bpm_est,
                                        patch_size=0, 
                                        RGB_LOW_HIGH_TH=(5,230),
                                        Skin_LOW_HIGH_TH=(5,230),
                                        pre_filt=True,
                                        post_filt=True,
                                        cuda=True, 
                                        verb=True)

# 打印每个时间戳的结果
print("估算出的BPM值:", bpmES)
print("对应的时间点 (秒):", timesES)

# 转换为简单的数值列表
bpm_values = [b.item() for b in bpmES]

# --- 计算统计指标 ---
mean_bpm = np.mean(bpm_values)     # 平均心率
std_bpm = np.std(bpm_values)       # 标准差，衡量心率的波动程度
min_bpm = np.min(bpm_values)       # 最小心率
max_bpm = np.max(bpm_values)       # 最大心率

print(f"视频片段的平均心率: {mean_bpm:.2f} BPM")
print(f"心率标准差: {std_bpm:.2f} BPM")
print(f"心率范围: 从 {min_bpm:.2f} BPM 到 {max_bpm:.2f} BPM")


try:
    # 打印或绘制估算结果
    if bpmES is not None and timesES is not None:

        # 为了方便绘图和计算，将 bpmES 转换为简单列表
        bpm_values = [b.item() for b in bpmES]

        # 使用 matplotlib 绘制结果图
        plt.figure(figsize=(12, 6))
        plt.plot(timesES, bpm_values, label='Estimated heart rate (BPM)', marker='o')
        plt.xlabel("time (sencond)")
        plt.ylabel("BPM")
        plt.title("Heart rate estimation result")
        plt.legend()
        plt.grid(True)
        
        # --- 关键修改在这里 ---
        # 将图像保存到文件，文件名可以自定义
        output_filename = 'heart_rate_plot.png'
        plt.savefig(output_filename)
        print(f"图像已成功保存到: {output_filename}")

    else:
        print("未能成功估算出BPM值，无法绘制图像。")

except Exception as e:
    print(f"处理或绘图时出错: {e}")
    print("请检查视频路径是否正确，以及视频中是否能检测到人脸。")

# ERRORS
RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(bvps, fps, bpmES, bpmGT, timesES, timesGT)
printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
displayErrors(bpmES, bpmGT, timesES, timesGT)