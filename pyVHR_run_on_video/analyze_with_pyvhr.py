import pyVHR as vhr
from pyVHR.analysis.pipeline import Pipeline
from pyVHR.plot.visualize import *
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import os
from datetime import datetime


# 全局参数
fps = 30

# --------------------------------------------------------------------------
# 函数1: 加载GT文件
# --------------------------------------------------------------------------
def load_gt_file(gt_path):
    """
    加载Ground Truth BPM数据
    
    Parameters:
    -----------
    gt_path : str
        预处理时生成的.npz文件路径(不含扩展名，例如Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT），直接使用代码拼接.npz并读取
    
    Returns:
    --------
    bpm, times, metadata
    """
    try:
        data = np.load(f"{gt_path}.npz", allow_pickle=True)
        bpm = data['bpm']
        times = data['times']
        metadata = data['metadata'].item() if 'metadata' in data else {}
        print(f"  ✓ 成功加载GT文件: {gt_path}.npz")
        print(f"    - BPM数量: {len(bpm)}")
        print(f"    - 时间范围: {times[0]:.2f}s - {times[-1]:.2f}s")
        print(f"    - BPM范围: {np.min(bpm):.2f} - {np.max(bpm):.2f}")
        return bpm, times, metadata
    except Exception as e:
        print(f"  ✗ 加载GT文件失败: {e}")
        return None, None, None

# --------------------------------------------------------------------------
# 函数2: 分析视频
# --------------------------------------------------------------------------
def analyze_video(video_path, pipe, video_id=1, wsize=16, 
                  roi_method='convexhull', method='cpu_CHROM', 
                  roi_approach='holistic', estimate='holistic'):
    """
    分析单个视频的心率特征
    """
    print(f"\n  → 正在分析视频...")
    
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

    if bpmES is None or timesES is None or len(bpmES) == 0:
        print(f"  ✗ 警告: 视频 {video_id} 未能成功估算出BPM值。")
        return None, None, None
    
    # 同时支持 ICA（多输出）和 POS/CHROM（单输出）
    bpm_values = []
    last_valid_bpm = None

    for b in bpmES:
        # 确保 b 是 numpy 数组
        b_arr = np.atleast_1d(b)
        
        if b_arr.size == 1:
            # 正常情况 (POS, CHROM 等单输出算法)
            val = b_arr.item()
        else:
            # ICA 情况：返回了多个分量 (通常是3个)
            # 策略：心率具有连续性，选择与上一帧结果跳变最小的那个分量
            if last_valid_bpm is not None:
                # 找到与上一次 BPM 差值绝对值最小的索引
                best_idx = np.argmin(np.abs(b_arr - last_valid_bpm))
                val = b_arr[best_idx]
            else:
                # 如果是第一帧，没有历史参考，则取中位数 (相对稳健)
                val = np.median(b_arr)
                
        bpm_values.append(val)
        last_valid_bpm = val

    mean_bpm = np.mean(bpm_values)
    std_bpm = np.std(bpm_values)
    min_bpm = np.min(bpm_values)
    max_bpm = np.max(bpm_values)

    print(f"\n  ✓ 视频分析完成:")
    print(f"    - BPM数量: {len(bpm_values)}")
    print(f"    - 时间范围: {timesES[0]:.2f}s - {timesES[-1]:.2f}s")
    print(f"    - 平均心率: {mean_bpm:.2f} BPM")
    print(f"    - 标准差: {std_bpm:.2f} BPM")
    print(f"    - 心率范围: {min_bpm:.2f} - {max_bpm:.2f} BPM")

    return bvps, timesES, bpm_values

# --------------------------------------------------------------------------
# 函数3: 绘制单个视频对比图
# --------------------------------------------------------------------------
def plot_comparison_single(timesES, bpmES, timesGT, bpmGT, video_name, output_path):
    """
    绘制单个视频的rPPG估计值与GT值的对比图
    """
    plt.figure(figsize=(14, 7))
    
    plt.plot(timesGT, bpmGT, label='Ground Truth (GT)', 
             marker='s', markersize=6, linestyle='-', linewidth=2.5, 
             color='#2ecc71', alpha=0.8)
    
    plt.plot(timesES, bpmES, label='rPPG Estimation (ES)', 
             marker='o', markersize=6, linestyle='--', linewidth=2.5, 
             color='#e74c3c', alpha=0.8)
    
    plt.xlabel("Time (seconds)", fontsize=13, fontweight='bold')
    plt.ylabel("BPM (Beats Per Minute)", fontsize=13, fontweight='bold')
    plt.title(f"Heart Rate Comparison: {video_name}", fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    mean_es = np.mean(bpmES)
    mean_gt = np.mean(bpmGT)
    textstr = f'Mean ES: {mean_es:.2f} BPM\nMean GT: {mean_gt:.2f} BPM\nDiff: {abs(mean_es-mean_gt):.2f} BPM'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
             fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {output_path}")
    plt.close()

# --------------------------------------------------------------------------
# 计算视频组平均误差
# --------------------------------------------------------------------------
def calculate_group_average_errors(video_errors):
    """
    计算视频组内所有视频的平均误差指标
    
    Parameters:
    -----------
    video_errors : list
        包含各个视频误差指标的列表
    
    Returns:
    --------
    dict : 平均误差指标
    """
    if not video_errors:
        return None
    
    avg_metrics = {
        'RMSE': np.mean([v['RMSE'] for v in video_errors]),
        'MAE': np.mean([v['MAE'] for v in video_errors]),
        'MAX': np.mean([v['MAX'] for v in video_errors]),
        'PCC': np.mean([v['PCC'] for v in video_errors]),
        'CCC': np.mean([v['CCC'] for v in video_errors]),
        'SNR': np.mean([v['SNR'] for v in video_errors])
    }
    
    # 计算标准差
    std_metrics = {
        'RMSE_std': np.std([v['RMSE'] for v in video_errors]),
        'MAE_std': np.std([v['MAE'] for v in video_errors]),
        'MAX_std': np.std([v['MAX'] for v in video_errors]),
        'PCC_std': np.std([v['PCC'] for v in video_errors]),
        'CCC_std': np.std([v['CCC'] for v in video_errors]),
        'SNR_std': np.std([v['SNR'] for v in video_errors])
    }
    
    return {**avg_metrics, **std_metrics}

# --------------------------------------------------------------------------
# 绘制视频组综合对比图
# --------------------------------------------------------------------------
def plot_group_comparison(group_results, group_name, output_path):
    """
    绘制视频组内所有视频的对比图
    """
    plt.figure(figsize=(16, 9))
    
    for idx, result in enumerate(group_results):
        color_es = plt.cm.tab10(idx * 2 % 10)
        color_gt = plt.cm.tab10((idx * 2 + 1) % 10)
        
        plt.plot(result['times'], result['bpms'], 
                label=f'{result["name"]} (ES)', 
                marker='o', markersize=4, linestyle='--', 
                linewidth=2, alpha=0.7, color=color_es)
        plt.plot(result['timesGT'], result['bpmGT'], 
                label=f'{result["name"]} (GT)', 
                marker='s', markersize=4, linestyle='-', 
                linewidth=2.5, alpha=0.7, color=color_gt)
    
    plt.xlabel("Time (seconds)", fontsize=13, fontweight='bold')
    plt.ylabel("BPM (Beats Per Minute)", fontsize=13, fontweight='bold')
    plt.title(f"Group Comparison: {group_name}", fontsize=15, fontweight='bold')
    plt.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 视频组对比图已保存: {output_path}")
    plt.close()

# --------------------------------------------------------------------------
# 新增函数: 绘制所有视频组的误差指标条形图对比
# --------------------------------------------------------------------------
def plot_groups_metrics_barchart(all_group_summaries, output_path):
    """
    将所有视频组的平均误差指标用条形图（带误差棒）进行对比
    """
    if not all_group_summaries:
        return

    # 提取组名和指标
    group_names = [summary['group_name'] for summary in all_group_summaries]
    
    # 定义需要绘制的指标
    metrics_to_plot = ['RMSE', 'MAE', 'SNR', 'PCC', 'CCC']
    
    # 设置画布 (2行3列，或者根据指标数量自适应，这里选 2x3 留空一个)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    x = np.arange(len(group_names))
    width = 0.5  # 条形图宽度
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # 提取平均值和标准差
        means = [summary['average_metrics'][metric] for summary in all_group_summaries]
        stds = [summary['average_metrics'][f'{metric}_std'] for summary in all_group_summaries]
        
        # 绘制条形图并添加误差棒 (yerr)
        bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.8, 
                      color=plt.cm.Set2(i), edgecolor='black')
        
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        
        # 处理X轴标签太长重叠的问题
        formatted_names = [name.replace(" - ", "\n") for name in group_names]
        ax.set_xticklabels(formatted_names, rotation=15, ha='right', fontsize=10)
        
        # 为不同类型的指标设置合适的Y轴标签
        if metric in ['RMSE', 'MAE']:
            ax.set_ylabel('BPM', fontsize=11)
        elif metric == 'SNR':
            ax.set_ylabel('dB', fontsize=11)
        else:
            ax.set_ylabel('Value (0-1)', fontsize=11)
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 在柱子上标出具体数值
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(means),
                    f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    # 隐藏最后一个多余的子图
    axes[5].axis('off')
    
    plt.suptitle('Comparison of Error Metrics Across Video Groups', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n ✓ 所有视频组指标对比条形图已保存: {output_path}")
    plt.close()

# --------------------------------------------------------------------------
#   绘制所有视频组的误差指标箱型图对比
# --------------------------------------------------------------------------
def plot_groups_metrics_boxplot(all_group_summaries, output_path):
    """
    将所有视频组的各个视频误差指标用箱型图进行分布对比
    """
    if not all_group_summaries:
        return

    group_names = [summary['group_name'] for summary in all_group_summaries]
    metrics_to_plot = ['RMSE', 'MAE', 'SNR', 'PCC', 'CCC']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # 核心：提取每个Group中，所有单独视频的特定指标数据，组成二维列表
        # 例如: [[group1_v1_rmse, group1_v2_rmse...], [group2_v1_rmse, ...]]
        data_to_plot = []
        for summary in all_group_summaries:
            metric_values = [video['metrics'][metric] for video in summary['individual_videos']]
            data_to_plot.append(metric_values)
            
        # 绘制箱型图
        # patch_artist=True 允许填充颜色
        # showmeans=True 会用三角形标出均值的位置，方便和中位数对比
        box = ax.boxplot(data_to_plot, patch_artist=True, showmeans=True,
                         meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8},
                         flierprops={"marker":"o", "markerfacecolor":"red", "alpha":0.5}) # 异常值标为红色
        
        # 为不同组的箱子涂上不同的颜色
        colors = plt.cm.Set2(np.linspace(0, 1, len(group_names)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        # 修改中位数的线条样式
        for median in box['medians']:
            median.set(color='black', linewidth=2)
            
        ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
        
        # 处理X轴标签
        formatted_names = [name.replace(" - ", "\n") for name in group_names]
        ax.set_xticks(range(1, len(group_names) + 1))
        ax.set_xticklabels(formatted_names, rotation=15, ha='right', fontsize=10)
        
        if metric in ['RMSE', 'MAE']:
            ax.set_ylabel('BPM', fontsize=11)
        elif metric == 'SNR':
            ax.set_ylabel('dB', fontsize=11)
        else:
            ax.set_ylabel('Value (0-1)', fontsize=11)
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 隐藏最后一个多余的子图
    axes[5].axis('off')
    
    # 添加图例说明
    import matplotlib.lines as mlines
    mean_line = mlines.Line2D([], [], color='white', marker='^', markeredgecolor='black', markersize=8, label='Mean (均值)')
    median_line = mlines.Line2D([], [], color='black', linewidth=2, label='Median (中位数)')
    outlier_dot = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='red', alpha=0.5, label='Outlier (异常值)')
    fig.legend(handles=[mean_line, median_line, outlier_dot], loc='lower right', fontsize=12, bbox_to_anchor=(0.95, 0.1))
    
    plt.suptitle('Distribution of Error Metrics Across Video Groups (Boxplots)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n ✓ 所有视频组指标分布箱型图已保存: {output_path}")
    plt.close()

# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ========== 配置区域 ==========
    # 定义视频组结构
    video_groups = [
        {
            'group_name': 'baseenv_baselineISP_VG',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_hyl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
                    'name': 'baseenv_baselineISP_hyl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_lj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
                    'name': 'baseenv_baselineISP_lj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_lxr_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
                    'name': 'baseenv_baselineISP_lxr'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_lzj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
                    'name': 'baseenv_baselineISP_lzj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_lzz_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
                    'name': 'baseenv_baselineISP_lzz'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_wyx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
                    'name': 'baseenv_baselineISP_wyx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_wzx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
                    'name': 'baseenv_baselineISP_wzx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_ycl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
                    'name': 'baseenv_baselineISP_ycl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_yjc_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
                    'name': 'baseenv_baselineISP_yjc'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_zbw_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
                    'name': 'baseenv_baselineISP_zbw'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/baseenv_baselineISP/raw_zxh_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
                    'name': 'baseenv_baselineISP_zxh'
                },
            ]
        },
        {
            'group_name': 'baseenv_0.8gammaISP_VG',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_hyl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_hyl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_lj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_lj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_lxr_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_lxr'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_lzj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_lzj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_lzz_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_lzz'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_wyx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_wyx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_wzx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_wzx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_ycl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_ycl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_yjc_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_yjc'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_zbw_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_zbw'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/0.8/raw_zxh_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
                    'name': 'baseenv_0.8gammaISP_zxh'
                },
            ]
        },
        {
            'group_name': 'baseenv_1.0gammaISP_VG',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_hyl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_hyl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_lj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_lj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_lxr_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_lxr'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_lzj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_lzj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_lzz_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_lzz'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_wyx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_wyx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_wzx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_wzx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_ycl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_ycl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_yjc_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_yjc'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_zbw_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_zbw'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.0/raw_zxh_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
                    'name': 'baseenv_1.0gammaISP_zxh'
                },
            ]
        },
        {
            'group_name': 'baseenv_3.0gammaISP_VG',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_hyl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_hyl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_lj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lxr_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_lxr'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lzj_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_lzj'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lzz_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_lzz'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_wyx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_wyx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_wzx_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_wzx'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_ycl_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_ycl'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_yjc_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_yjc'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_zbw_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_zbw'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_zxh_output_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
                    'name': 'baseenv_3.0gammaISP_zxh'
                },
            ]
        },
        # {
        #     'group_name': 'baseenv_1.8gammaISP_VG',
        #     'videos': [
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_hyl_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_hyl'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_lj_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_lj'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_lxr_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_lxr'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_lzj_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_lzj'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_lzz_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_lzz'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_wyx_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_wyx'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_wzx_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_wzx'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_ycl_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_ycl'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_yjc_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_yjc'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_zbw_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_zbw'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/1.8/raw_zxh_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
        #             'name': 'baseenv_1.8gammaISP_zxh'
        #         },
        #     ]
        # },{
        #     'group_name': 'baseenv_3.0gammaISP_VG',
        #     'videos': [
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_hyl_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_hyl/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_hyl'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lj_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lj/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_lj'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lxr_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lxr/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_lxr'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lzj_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lzj/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_lzj'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_lzz_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_lzz/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_lzz'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_wyx_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_wyx/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_wyx'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_wzx_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_wzx/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_wzx'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_ycl_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_ycl/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_ycl'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_yjc_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_yjc/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_yjc'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_zbw_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_zbw/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_zbw'
        #         },
        #         {
        #             'video_path': 'Data_for_pyVHR/isp_output_Video/sensitivity_analysis/baseenv/gamma/3.0/raw_zxh_output.mkv',
        #             'gt_path': 'Data_for_pyVHR/gt_data/gt_zxh/bpms_times_GT',
        #             'name': 'baseenv_3.0gammaISP_zxh'
        #         },
        #     ]
        # },
        # 可以继续添加更多视频组...
    ]
    
    rppg_method = 'cpu_CHROM'
    window_size = 16
    roi_method = 'convexhull'
    roi_approach = 'holistic'
    estimate = 'holistic'
    output_dir = 'rPPGanalyze_res_plots/sensitivity_analysis/gamma/cpu_CHROM_v1'
 
    analysis_params = {
        'rppg_method': rppg_method,
        'window_size': window_size,
        'roi_method': roi_method,
        'roi_approach': roi_approach,
        'estimate': estimate,
        'patch_size': 40,
        'RGB_LOW_HIGH_TH': (75, 230),
        'Skin_LOW_HIGH_TH': (75, 230),
        'pre_filt': True,
        'post_filt': True,
        'cuda': True
    }
    # =============================
    
    print(f"\n{'='*80}")
    print(f"pyVHR 视频组批量分析与GT对比系统")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"视频组数量: {len(video_groups)}")
    
    total_videos = sum(len(group['videos']) for group in video_groups)
    print(f"待处理视频总数: {total_videos}")

    print(f"\n--- 本次运行分析参数 ---")
    print(json.dumps(analysis_params, indent=2))
    print(f"{'='*80}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {output_dir}\n")
    
    pipe = Pipeline()
    
    # 存储所有视频组的结果
    all_group_summaries = []
    
    # 循环处理每个视频组
    for group_idx, video_group in enumerate(video_groups):
        group_name = video_group['group_name']
        videos = video_group['videos']
        
        print(f"\n{'='*80}")
        print(f"处理视频组 [{group_idx+1}/{len(video_groups)}]: {group_name}")
        print(f"该组包含 {len(videos)} 个视频")
        print(f"{'='*80}")
        
        # 为该视频组创建子目录
        group_output_dir = os.path.join(output_dir, f'group_{group_idx+1}_{group_name.replace(" ", "_")}')
        os.makedirs(group_output_dir, exist_ok=True)

        # 为单个视频的对比图和误差图创建专属的子文件夹
        indiv_plots_dir = os.path.join(group_output_dir, 'singleVideo_rppg_analysis')
        os.makedirs(indiv_plots_dir, exist_ok=True)
        
        group_results = []
        group_errors = []
        
        # 循环处理该组内的每个视频
        for video_idx, config in enumerate(videos):
            print(f"\n{'-'*70}")
            print(f"视频进度: [{video_idx+1}/{len(videos)}] - {config['name']}")
            print(f"{'-'*70}")
            
            video_path = config['video_path']
            gt_path = config['gt_path']
            video_name = config['name']
            
            # 步骤1: 加载GT
            print(f"\n[步骤 1/4] 加载Ground Truth数据...")
            bpmGT, timesGT, metadata = load_gt_file(gt_path)
            
            if bpmGT is None:
                print(f"  ✗ 跳过视频 {video_name}: 无法加载GT数据\n")
                continue
            
            # 步骤2: 分析视频
            print(f"\n[步骤 2/4] 使用pyVHR分析视频...")
            bvps, timesES, bpmES = analyze_video(
                video_path, pipe, video_id=video_idx+1, 
                method=rppg_method,
                wsize=window_size,
                roi_method=roi_method,
                roi_approach=roi_approach,
                estimate=estimate
            )
            
            if timesES is None or bpmES is None:
                print(f"  ✗ 跳过视频 {video_name}: 分析失败\n")
                continue
            
            bpmES_array = np.array(bpmES)
            timesES_array = np.array(timesES)
            
            # 步骤3: 绘制单个视频对比图
            print(f"\n[步骤 3/4] 生成对比图像...")
            comparison_plot_path = os.path.join(indiv_plots_dir, f'{video_name}_heartRate_comparison.png')
            plot_comparison_single(timesES_array, bpmES_array, timesGT, bpmGT, 
                                  video_name, comparison_plot_path)
            
            # 步骤4: 误差分析
            print(f"\n[步骤 4/4] 计算误差指标...")
            try:
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(
                    bvps, fps, bpmES_array, bpmGT, timesES_array, timesGT
                )
                
                print(f"\n  --- {video_name} 的误差指标 ---")
                printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)
                
                error_metrics = {
                    'RMSE': float(RMSE) if hasattr(RMSE, 'item') else float(RMSE),
                    'MAE': float(MAE) if hasattr(MAE, 'item') else float(MAE),
                    'MAX': float(MAX) if hasattr(MAX, 'item') else float(MAX),
                    'PCC': float(PCC) if hasattr(PCC, 'item') else float(PCC),
                    'CCC': float(CCC) if hasattr(CCC, 'item') else float(CCC),
                    'SNR': float(SNR) if hasattr(SNR, 'item') else float(SNR)
                }
                
                group_errors.append(error_metrics)
                
                # 绘制误差分析图
                error_plot_path = os.path.join(indiv_plots_dir, f'{video_name}_rppgError_analysis.png')
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                
                timesGT_array = np.array(timesGT)
                bpmGT_array = np.array(bpmGT)
                timesES_plot = np.array(timesES_array)
                bpmES_plot = np.array(bpmES_array)
                
                # 单个视频分析子图1: BPM对比
                ax1.plot(timesGT_array, bpmGT_array, 'g-', label='Ground Truth', 
                        linewidth=2.5, marker='s', markersize=5, alpha=0.8)
                ax1.plot(timesES_plot, bpmES_plot, 'r--', label='Estimation', 
                        linewidth=2.5, marker='o', markersize=5, alpha=0.8)
                ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('BPM', fontsize=12, fontweight='bold')
                ax1.set_title(f'BPM Comparison: {video_name}', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # 单个视频分析子图2: 绝对误差
                if len(timesGT_array) > 1 and len(timesES_plot) > 1:
                    try:
                        f_interp = interp1d(timesGT_array, bpmGT_array, 
                                           kind='linear', fill_value='extrapolate')
                        bpmGT_interp = f_interp(timesES_plot)
                        abs_errors = np.abs(bpmES_plot - bpmGT_interp)
                        
                        ax2.plot(timesES_plot, abs_errors, 'b-', linewidth=2.5, 
                                marker='o', markersize=5, alpha=0.8, label='Absolute Error')
                        ax2.axhline(y=error_metrics['MAE'], color='r', linestyle='--', 
                                   linewidth=2, label=f'MAE = {error_metrics["MAE"]:.2f}')
                        ax2.fill_between(timesES_plot, 0, abs_errors, alpha=0.2, color='blue')
                        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Absolute Error (BPM)', fontsize=12, fontweight='bold')
                        ax2.set_title('Absolute Error Over Time', fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=11)
                        ax2.grid(True, alpha=0.3, linestyle='--')
                    except Exception as interp_error:
                        print(f"  ! 插值计算出错: {interp_error}")
                        ax2.text(0.5, 0.5, 'Error calculating interpolation', 
                                ha='center', va='center', transform=ax2.transAxes)
                
                plt.tight_layout()
                plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ 误差分析图已保存: {error_plot_path}")
                plt.close()
                
            except Exception as e:
                import traceback
                print(f"  ✗ 误差分析出错: {e}")
                traceback.print_exc()
            
            group_results.append({
                'name': video_name,
                'times': timesES_array,
                'bpms': bpmES_array,
                'timesGT': timesGT,
                'bpmGT': bpmGT
            })
            
            print(f"\n✓ 视频 {video_name} 处理完成！")
        
        # 计算该视频组的平均误差
        if group_errors:
            avg_errors = calculate_group_average_errors(group_errors)
            
            print(f"\n{'='*80}")
            print(f"视频组 [{group_name}] 平均误差统计")
            print(f"{'='*80}")
            print(f"指标\t\t平均值\t\t标准差")
            print(f"{'-'*80}")
            print(f"RMSE:\t\t{avg_errors['RMSE']:.3f}\t\t{avg_errors['RMSE_std']:.3f}")
            print(f"MAE:\t\t{avg_errors['MAE']:.3f}\t\t{avg_errors['MAE_std']:.3f}")
            print(f"MAX:\t\t{avg_errors['MAX']:.3f}\t\t{avg_errors['MAX_std']:.3f}")
            print(f"PCC:\t\t{avg_errors['PCC']:.3f}\t\t{avg_errors['PCC_std']:.3f}")
            print(f"CCC:\t\t{avg_errors['CCC']:.3f}\t\t{avg_errors['CCC_std']:.3f}")
            print(f"SNR:\t\t{avg_errors['SNR']:.3f}\t\t{avg_errors['SNR_std']:.3f}")
            print(f"{'='*80}\n")
            
            # 保存视频组摘要
            group_summary = {
                'group_name': group_name,
                'video_count': len(group_results),
                'parameters': analysis_params,
                'average_metrics': avg_errors,
                'individual_videos': [
                    {'name': group_results[i]['name'], 'metrics': group_errors[i]}
                    for i in range(len(group_errors))
                ]
            }
            all_group_summaries.append(group_summary)
            
            # 保存该组的JSON文件
            group_json_path = os.path.join(group_output_dir, 'group_summary.json')
            with open(group_json_path, 'w') as f:
                json.dump(group_summary, f, indent=4)
            print(f"✓ 视频组摘要已保存: {group_json_path}")
            
            # 单个视频组综合对比图
            if group_results:
                group_comparison_path = os.path.join(group_output_dir, 'group_comparison.png')
                plot_group_comparison(group_results, group_name, group_comparison_path)
    
    # 保存所有视频组的总摘要
    if all_group_summaries:
        total_summary_path = os.path.join(output_dir, 'all_groups_summary.json')
        with open(total_summary_path, 'w') as f:
            json.dump(all_group_summaries, f, indent=4)
        print(f"\n✓ 所有视频组总摘要已保存: {total_summary_path}")
        
        # 打印所有视频组的对比表格
        print(f"\n{'='*100}")
        print(f"所有视频组平均误差对比 (方法: {rppg_method})")
        print(f"{'='*100}")
        print(f"{'视频组':<30} {'RMSE':<10} {'MAE':<10} {'MAX':<10} {'PCC':<10} {'CCC':<10} {'SNR':<10}")
        print(f"{'-'*100}")
        for summary in all_group_summaries:
            metrics = summary['average_metrics']
            print(f"{summary['group_name']:<30} {metrics['RMSE']:<10.3f} {metrics['MAE']:<10.3f} "
                  f"{metrics['MAX']:<10.3f} {metrics['PCC']:<10.3f} {metrics['CCC']:<10.3f} {metrics['SNR']:<10.3f}")
        print(f"{'='*100}\n")
    
        # 1. 绘制所有视频组的平均误差条形图
        final_barchart_path = os.path.join(output_dir, 'all_groups_barchart_of_average_errors.png')
        plot_groups_metrics_barchart(all_group_summaries, final_barchart_path)

        # 2. 绘制学术级箱型图分布图
        final_boxplot_path = os.path.join(output_dir, 'all_groups_metrics_boxplot.png')
        plot_groups_metrics_boxplot(all_group_summaries, final_boxplot_path)

    print(f"\n{'='*80}")
    print(f"所有处理完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功处理视频组数: {len(all_group_summaries)}/{len(video_groups)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")


# ============================================================================
# 自动化集成接口函数（供 automation_pipeline.py 调用）
# ============================================================================

def run_pyvhr_analysis(video_groups: list, analysis_params: dict, output_dir: str) -> dict:
    """
    pyVHR 分析封装函数，供自动化脚本调用

    Parameters:
    -----------
    video_groups : list
        视频组配置列表
    analysis_params : dict
        pyVHR 分析参数（rppg_method, window_size, roi_method 等）
    output_dir : str
        分析结果输出目录

    Returns:
    --------
    dict : 包含所有视频组的分析结果摘要
    """

    print(f"\n{'='*80}")
    print(f"pyVHR 视频组批量分析系统")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"视频组数量: {len(video_groups)}")

    total_videos = sum(len(group['videos']) for group in video_groups)
    print(f"待处理视频总数: {total_videos}")

    print(f"\n--- 本次运行分析参数 ---")
    print(json.dumps(analysis_params, indent=2))
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {output_dir}\n")

    pipe = Pipeline()
    all_group_summaries = []
    # 循环处理每个视频组
    for group_idx, video_group in enumerate(video_groups):
        group_name = video_group['group_name']
        videos = video_group['videos']

        print(f"\n{'='*80}")
        print(f"处理视频组 [{group_idx+1}/{len(video_groups)}]: {group_name}")
        print(f"该组包含 {len(videos)} 个视频")
        print(f"{'='*80}")

        group_output_dir = os.path.join(output_dir, f'group_{group_idx+1}_{group_name.replace(" ", "_")}')
        os.makedirs(group_output_dir, exist_ok=True)

        indiv_plots_dir = os.path.join(group_output_dir, 'singleVideo_rppg_analysis')
        os.makedirs(indiv_plots_dir, exist_ok=True)

        group_results = []
        group_errors = []

        # 循环处理该组内的每个视频
        for video_idx, config in enumerate(videos):
            print(f"\n{'-'*70}")
            print(f"视频进度: [{video_idx+1}/{len(videos)}] - {config['name']}")
            print(f"{'-'*70}")

            video_path = config['video_path']
            gt_path = config['gt_path']
            video_name = config['name']

            # 加载 GT
            print(f"\n[步骤 1/4] 加载Ground Truth数据...")
            bpmGT, timesGT, metadata = load_gt_file(gt_path)

            if bpmGT is None:
                print(f"  ✗ 跳过视频 {video_name}: 无法加载GT数据\n")
                continue

            # 分析视频
            print(f"\n[步骤 2/4] 使用pyVHR分析视频...")
            bvps, timesES, bpmES = analyze_video(
                video_path, pipe, video_id=video_idx+1,
                method=analysis_params['rppg_method'],
                wsize=analysis_params['window_size'],
                roi_method=analysis_params['roi_method'],
                roi_approach=analysis_params['roi_approach'],
                estimate=analysis_params['estimate']
            )

            if timesES is None or bpmES is None:
                print(f"  ✗ 跳过视频 {video_name}: 分析失败\n")
                continue

            bpmES_array = np.array(bpmES)
            timesES_array = np.array(timesES)

            # 绘制对比图
            print(f"\n[步骤 3/4] 生成对比图像...")
            comparison_plot_path = os.path.join(indiv_plots_dir, f'{video_name}_heartRate_comparison.png')
            plot_comparison_single(timesES_array, bpmES_array, timesGT, bpmGT, video_name, comparison_plot_path)

            # 误差分析
            print(f"\n[步骤 4/4] 计算误差指标...")
            try:
                RMSE, MAE, MAX, PCC, CCC, SNR = getErrors(
                    bvps, fps, bpmES_array, bpmGT, timesES_array, timesGT
                )

                print(f"\n  --- {video_name} 的误差指标 ---")
                printErrors(RMSE, MAE, MAX, PCC, CCC, SNR)

                error_metrics = {
                    'RMSE': float(RMSE) if hasattr(RMSE, 'item') else float(RMSE),
                    'MAE': float(MAE) if hasattr(MAE, 'item') else float(MAE),
                    'MAX': float(MAX) if hasattr(MAX, 'item') else float(MAX),
                    'PCC': float(PCC) if hasattr(PCC, 'item') else float(PCC),
                    'CCC': float(CCC) if hasattr(CCC, 'item') else float(CCC),
                    'SNR': float(SNR) if hasattr(SNR, 'item') else float(SNR)
                }

                group_errors.append(error_metrics)

                # 绘制误差分析图（简化版，复用现有逻辑）
                error_plot_path = os.path.join(indiv_plots_dir, f'{video_name}_rppgError_analysis.png')
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                timesGT_array = np.array(timesGT)
                bpmGT_array = np.array(bpmGT)

                ax1.plot(timesGT_array, bpmGT_array, 'g-', label='Ground Truth',
                        linewidth=2.5, marker='s', markersize=5, alpha=0.8)
                ax1.plot(timesES_array, bpmES_array, 'r--', label='Estimation',
                        linewidth=2.5, marker='o', markersize=5, alpha=0.8)
                ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('BPM', fontsize=12, fontweight='bold')
                ax1.set_title(f'BPM Comparison: {video_name}', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--')

                if len(timesGT_array) > 1 and len(timesES_array) > 1:
                    try:
                        f_interp = interp1d(timesGT_array, bpmGT_array, kind='linear', fill_value='extrapolate')
                        bpmGT_interp = f_interp(timesES_array)
                        abs_errors = np.abs(bpmES_array - bpmGT_interp)

                        ax2.plot(timesES_array, abs_errors, 'b-', linewidth=2.5,
                                marker='o', markersize=5, alpha=0.8, label='Absolute Error')
                        ax2.axhline(y=error_metrics['MAE'], color='r', linestyle='--',
                                   linewidth=2, label=f'MAE = {error_metrics["MAE"]:.2f}')
                        ax2.fill_between(timesES_array, 0, abs_errors, alpha=0.2, color='blue')
                        ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Absolute Error (BPM)', fontsize=12, fontweight='bold')
                        ax2.set_title('Absolute Error Over Time', fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=11)
                        ax2.grid(True, alpha=0.3, linestyle='--')
                    except Exception:
                        pass

                plt.tight_layout()
                plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ 误差分析图已保存: {error_plot_path}")
                plt.close()

            except Exception as e:
                print(f"  ✗ 误差分析出错: {e}")

            group_results.append({
                'name': video_name,
                'times': timesES_array,
                'bpms': bpmES_array,
                'timesGT': timesGT,
                'bpmGT': bpmGT
            })

            print(f"\n✓ 视频 {video_name} 处理完成！")

        # 计算该视频组的平均误差
        if group_errors:
            avg_errors = calculate_group_average_errors(group_errors)

            print(f"\n{'='*80}")
            print(f"视频组 [{group_name}] 平均误差统计")
            print(f"{'='*80}")
            print(f"RMSE: {avg_errors['RMSE']:.3f} ± {avg_errors['RMSE_std']:.3f}")
            print(f"MAE: {avg_errors['MAE']:.3f} ± {avg_errors['MAE_std']:.3f}")
            print(f"PCC: {avg_errors['PCC']:.3f} ± {avg_errors['PCC_std']:.3f}")
            print(f"{'='*80}\n")

            group_summary = {
                'group_name': group_name,
                'video_count': len(group_results),
                'parameters': analysis_params,
                'average_metrics': avg_errors,
                'individual_videos': [
                    {'name': group_results[i]['name'], 'metrics': group_errors[i]}
                    for i in range(len(group_errors))
                ]
            }
            all_group_summaries.append(group_summary)

            group_json_path = os.path.join(group_output_dir, 'group_summary.json')
            with open(group_json_path, 'w') as f:
                json.dump(group_summary, f, indent=4)
            print(f"✓ 视频组摘要已保存: {group_json_path}")

            if group_results:
                group_comparison_path = os.path.join(group_output_dir, 'group_comparison.png')
                plot_group_comparison(group_results, group_name, group_comparison_path)

    # 保存所有视频组的总摘要
    if all_group_summaries:
        total_summary_path = os.path.join(output_dir, 'all_groups_summary.json')
        with open(total_summary_path, 'w') as f:
            json.dump(all_group_summaries, f, indent=4)
        print(f"\n✓ 所有视频组总摘要已保存: {total_summary_path}")

        print(f"\n{'='*100}")
        print(f"所有视频组平均误差对比")
        print(f"{'='*100}")
        for summary in all_group_summaries:
            metrics = summary['average_metrics']
            print(f"{summary['group_name']:<30} RMSE:{metrics['RMSE']:.3f} MAE:{metrics['MAE']:.3f} PCC:{metrics['PCC']:.3f}")
        print(f"{'='*100}\n")

        final_barchart_path = os.path.join(output_dir, 'all_groups_barchart_of_average_errors.png')
        plot_groups_metrics_barchart(all_group_summaries, final_barchart_path)

        final_boxplot_path = os.path.join(output_dir, 'all_groups_metrics_boxplot.png')
        plot_groups_metrics_boxplot(all_group_summaries, final_boxplot_path)

    print(f"\n{'='*80}")
    print(f"pyVHR 分析完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功处理视频组数: {len(all_group_summaries)}/{len(video_groups)}")
    print(f"{'='*80}\n")

    return {
        'all_group_summaries': all_group_summaries,
        'output_dir': output_dir,
        'total_groups': len(all_group_summaries),
        'total_videos': total_videos
    }
