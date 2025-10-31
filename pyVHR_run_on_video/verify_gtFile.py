import numpy as np
import matplotlib.pyplot as plt

# 将 'ground_truth.txt' 替换为实际文件路径
file_path = 'Data_for_pyVHR/UBFC2/subject3/ground_truth.txt'

try:
    with open(file_path, 'r') as f:
        # readlines() 会按真实的换行符读取
        lines = f.readlines()

    print(f"文件中的总行数: {len(lines)}")

    if len(lines) >= 3:
        print("\n--- 数据预览 ---")
        # .split() 会按空格分割，[:5] 表示只看前5个
        print("第一行 (BVP信号) 的前5个数据:", lines[0].split()[:5])
        print("第二行 (心率值) 的前5个数据:", lines[1].split()[:5])
        print("第三行 (时间戳) 的前5个数据:", lines[2].split()[:5])

except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。")


# 验证自己拍摄的GT文件
data_sig = np.load('Data_for_pyVHR/HK-PPG-COM7_sig.npy')
data_ts =  np.load('Data_for_pyVHR/HK-PPG-COM7_ts.npy')

# 1. 预览部分数据
print("\n--- 实拍数据预览 ---")

print("sig信号的数组形状：", data_sig.shape)  
print("sig信号的数据类型：", data_sig.dtype)   
print("sig部分数据：", data_sig[: 100])  

print("ts信号的数组形状：", data_ts.shape)   
print("ts信号的数据类型：", data_ts.dtype)   
print("ts部分数据：", data_ts[: 100])   


print("--- 检查时间戳 (ts) ---")

# 2. 检查时间间隔 (这非常重要)
#    使用 np.diff() 计算相邻元素之间的差异
time_diffs = np.diff(data_ts)

print(f"前 10 个时间戳的差异 (采样间隔): \n{time_diffs[:10]}")

# 3. 计算平均采样率 (Fs) 和总时长
if len(time_diffs) > 0:
    mean_interval = np.mean(time_diffs)
    avg_fs = 1.0 / mean_interval
    duration = data_ts[-1] - data_ts[0]
    print(f"\n总时长: {duration:.2f} 秒")
    print(f"平均采样间隔: {mean_interval:.6f} 秒")
    print(f"估算采样率 (Fs): {avg_fs:.2f} Hz")
else:
    print("时间戳数组太短，无法计算差异。")


print("\n--- 检查信号 (sig) ---")

# 4. 检查信号的统计数据，看它是否全为0
print(f"信号最小值: {np.min(data_sig)}")
print(f"信号最大值: {np.max(data_sig)}")
print(f"信号平均值: {np.mean(data_sig)}")

# 5. 根据GT信号绘制信号波形
try:
    print("\n--- 正在尝试绘制信号图并保存 ---")
    
    #  创建一个图窗
    plt.figure(figsize=(15, 5))

    #  绘制信号：使用 ts 作为 x 轴 (时间), sig 作为 y 轴 (信号)
    plt.plot(data_ts, data_sig)

    #  添加标题和标签
    plt.xlabel("时间 (Unix 时间戳)")
    plt.ylabel("PPG 信号幅值")
    plt.title("红外脉搏 (PPG) 信号波形 (总时长 60 秒)")
    plt.grid(True) # 添加网格

    # 将图像保存到文件
    output_filename = "ppg_signal_plot.png"
    plt.savefig(output_filename)
    
    print(f"\n[成功] 图像已保存到文件: {output_filename}")

except Exception as e:
    print(f"\n[失败] 绘制图像时出错: {e}")
    print("请确保你已正确安装 matplotlib 库 (pip install matplotlib)")