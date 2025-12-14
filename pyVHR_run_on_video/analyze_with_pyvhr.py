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
        GT文件路径(不含扩展名)，代码读取.npz的文件
    
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
# 新增函数: 计算视频组平均误差
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
# 新增函数: 绘制视频组综合对比图
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
# 主程序
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ========== 配置区域 ==========
    # 定义视频组结构
    video_groups = [
        {
            'group_name': 'Video Group 1 - ISP(0)',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/output_video_1_isp(0)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(0)'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(1)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(0)'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(2)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(0)'
                },
            ]
        },
        {
            'group_name': 'Video Group 2 - Advanced ISP(2)',
            'videos': [
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(3)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(3)'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(4)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(4)'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(5)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(5)'
                },
                {
                    'video_path': 'Data_for_pyVHR/isp_output_Video/rawdenoise/Video_1_bin_and_space/output_video_1_isp(6)_8bit.mkv',
                    'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
                    'name': 'Video 1 - isp(6)'
                },
            ]
        },
        # 可以继续添加更多视频组...
    ]
    
    rppg_method = 'cpu_CHROM'
    window_size = 16
    roi_method = 'convexhull'
    roi_approach = 'holistic'
    estimate = 'holistic'
    output_dir = 'analyze_res_plots/group_based_analysis/rppg_CHROM'
 
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
            
            # 步骤3: 绘制对比图
            print(f"\n[步骤 3/4] 生成对比图像...")
            comparison_plot_path = os.path.join(group_output_dir, f'{video_name}_comparison.png')
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
                error_plot_path = os.path.join(group_output_dir, f'{video_name}_error_analysis.png')
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                
                timesGT_array = np.array(timesGT)
                bpmGT_array = np.array(bpmGT)
                timesES_plot = np.array(timesES_array)
                bpmES_plot = np.array(bpmES_array)
                
                # 子图1: BPM对比
                ax1.plot(timesGT_array, bpmGT_array, 'g-', label='Ground Truth', 
                        linewidth=2.5, marker='s', markersize=5, alpha=0.8)
                ax1.plot(timesES_plot, bpmES_plot, 'r--', label='Estimation', 
                        linewidth=2.5, marker='o', markersize=5, alpha=0.8)
                ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('BPM', fontsize=12, fontweight='bold')
                ax1.set_title(f'BPM Comparison: {video_name}', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=11)
                ax1.grid(True, alpha=0.3, linestyle='--')
                
                # 子图2: 绝对误差
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
            
            # 绘制视频组综合对比图
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
    
    print(f"\n{'='*80}")
    print(f"所有处理完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功处理视频组数: {len(all_group_summaries)}/{len(video_groups)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")