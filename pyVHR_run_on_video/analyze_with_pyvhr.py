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
        GT文件路径(不含扩展名)
    
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
def analyze_video(video_path, pipe, video_id=1, wsize=8, 
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
    
    bpm_values = [b.item() for b in bpmES]

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
# 主程序
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ========== 配置区域 ==========
    video_configs = [
        {
            'video_path': 'Data_for_pyVHR/isp_output_Video/output_video_data_1_8bit.mkv',
            'gt_path': 'Data_for_pyVHR/gt_data/raw16_1_gtData/HK-PPG-COM7_GT',
            'name': 'Video 1 (raw16_1)'
        },
        # 添加更多视频...
    ]
    
    rppg_method = 'cpu_CHROM'
    window_size = 8
    roi_method = 'convexhull'
    roi_approach = 'holistic'
    estimate = 'holistic'
    output_dir = 'analyze_res_plots/rppg_alg_cmp/Video_1-cpu_CHROM'
 
    #  这将被保存到 JSON 文件中
    analysis_params = {
        'rppg_method': rppg_method,
        'window_size': window_size,
        'roi_method': roi_method,
        'roi_approach': roi_approach,
        'estimate': estimate,
        # --- 以下参数来自 'analyze_video' 函数内部硬编码 ---
        'patch_size': 40,
        'RGB_LOW_HIGH_TH': (75, 230),
        'Skin_LOW_HIGH_TH': (75, 230),
        'pre_filt': True,
        'post_filt': True,
        'cuda': True
    }
    # =============================
    
    print(f"\n{'='*70}")
    print(f"pyVHR 批量视频分析与GT对比系统")
    print(f"{'='*70}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待处理视频数量: {len(video_configs)}")
    #print(f"rPPG方法: {rppg_method}")
    #print(f"窗口大小: {window_size}秒")

    print(f"\n--- 本次运行分析参数 ---")
    # 使用json打印字典，确保(75, 230)这样的元组也能正确显示
    print(json.dumps(analysis_params, indent=2))
    print(f"{'='*70}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录已创建: {output_dir}\n")
    
    all_results = []
    error_summary = []
    pipe = Pipeline() 
    
    # 循环处理每个视频
    for i, config in enumerate(video_configs):
        print(f"\n{'='*70}")
        print(f"处理进度: [{i+1}/{len(video_configs)}] - {config['name']}")
        print(f"{'='*70}")
        
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
            video_path, pipe, video_id=i+1, 
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
        comparison_plot_path = os.path.join(output_dir, f'{video_name}_comparison.png')
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
            
            # 确保转换为标量
            mae_value = float(MAE) if hasattr(MAE, 'item') else float(MAE)

            error_metrics = {
                'RMSE': float(RMSE) if hasattr(RMSE, 'item') else float(RMSE),
                'MAE': mae_value,
                'MAX': float(MAX) if hasattr(MAX, 'item') else float(MAX),
                'PCC': float(PCC) if hasattr(PCC, 'item') else float(PCC),
                'CCC': float(CCC) if hasattr(CCC, 'item') else float(CCC),
                'SNR': float(SNR) if hasattr(SNR, 'item') else float(SNR)
            }
            
            #   将参数和误差指标一起存入
            error_summary.append({
                'name': video_name,
                'parameters': analysis_params,  # <--- rppg参数
                'metrics': error_metrics       # <--- 嵌套指标
            })
                    
            # 使用matplotlib绘制误差分析图（避免Plotly的HTML输出）
            error_plot_path = os.path.join(output_dir, f'{video_name}_error_analysis.png')
            
            # 创建简单的误差对比图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # 确保数据是numpy数组
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
                    # 插值GT到ES的时间点
                    f_interp = interp1d(timesGT_array, bpmGT_array, 
                                       kind='linear', fill_value='extrapolate')
                    bpmGT_interp = f_interp(timesES_plot)
                    abs_errors = np.abs(bpmES_plot - bpmGT_interp)
                    
                    ax2.plot(timesES_plot, abs_errors, 'b-', linewidth=2.5, 
                            marker='o', markersize=5, alpha=0.8, label='Absolute Error')
                    ax2.axhline(y=mae_value, color='r', linestyle='--', 
                               linewidth=2, label=f'MAE = {mae_value:.2f}')
                    ax2.fill_between(timesES_plot, 0, abs_errors, alpha=0.2, color='blue')
                    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Absolute Error (BPM)', fontsize=12, fontweight='bold')
                    ax2.set_title('Absolute Error Over Time', fontsize=14, fontweight='bold')
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.3, linestyle='--')
                except Exception as interp_error:
                    print(f"  ! 插值计算出错: {interp_error}, 跳过误差图")
                    ax2.text(0.5, 0.5, 'Error calculating interpolation', 
                            ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ 误差分析图已保存: {error_plot_path}")
            plt.close()
            
        except Exception as e:
            import traceback
            print(f"  ✗ 误差分析出错: {e}")
            print(f"  详细错误信息:")
            traceback.print_exc()
        
        all_results.append({
            'id': f'Video {i+1}',
            'name': video_name,
            'times': timesES_array,
            'bpms': bpmES_array,
            'timesGT': timesGT,
            'bpmGT': bpmGT
        })
        
        print(f"\n✓ 视频 {video_name} 处理完成！")

    # 统一绘图
    print(f"\n{'='*70}")
    print(f"生成综合对比图...")
    print(f"{'='*70}\n")
    
    try:
        if all_results:
            plt.figure(figsize=(16, 9))

            for idx, result in enumerate(all_results):
                color_es = plt.cm.Set1(idx * 2)
                color_gt = plt.cm.Set1(idx * 2 + 1)
                
                plt.plot(result['times'], result['bpms'], 
                        label=f'{result["name"]} (ES)', 
                        marker='o', markersize=5, linestyle='--', 
                        linewidth=2, alpha=0.8, color=color_es)
                plt.plot(result['timesGT'], result['bpmGT'], 
                        label=f'{result["name"]} (GT)', 
                        marker='s', markersize=5, linestyle='-', 
                        linewidth=2.5, alpha=0.8, color=color_gt)

            plt.xlabel("Time (seconds)", fontsize=13, fontweight='bold')
            plt.ylabel("BPM (Beats Per Minute)", fontsize=13, fontweight='bold')
            plt.title("Comparison of All Videos: rPPG Estimation vs Ground Truth", 
                     fontsize=15, fontweight='bold')
            plt.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
            plt.grid(True, alpha=0.3, linestyle='--')
            
            output_filename = os.path.join(output_dir, 'all_videos_comparison.png')
            plt.tight_layout()
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"✓ 综合对比图已保存: {output_filename}")
            plt.close()

        else:
            print("✗ 未能成功估算出任何视频的BPM值")

    except Exception as e:
        print(f"✗ 处理或绘图时出错: {e}")
    
    # 保存误差摘要
    if error_summary:
        summary_path = os.path.join(output_dir, 'error_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(error_summary, f, indent=4)
        print(f"✓ 误差摘要已保存: {summary_path}")
        
        print(f"\n{'='*70}")
        print(f"误差分析摘要 (方法: {rppg_method})")
        print(f"{'='*70}")
        print(f"{'视频名称':<20} {'RMSE':<8} {'MAE':<8} {'MAX':<8} {'PCC':<8} {'CCC':<8}")
        print(f"{'-'*70}")
        for item in error_summary:
            metrics = item['metrics'] # 从 'metrics' 键中获取指标
            print(f"{item['name']:<20} {metrics['RMSE']:<8.2f} {metrics['MAE']:<8.2f} "
                  f"{metrics['MAX']:<8.2f} {metrics['PCC']:<8.3f} {metrics['CCC']:<8.3f}")
    
    print(f"\n{'='*70}")
    print(f"所有处理完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功处理视频数: {len(all_results)}/{len(video_configs)}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")