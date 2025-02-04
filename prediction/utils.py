"""prediction/utils"""
import logging
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import seaborn as sns

def plot_results_plus_origin(true_ddg, predictions, save_path, metrics=None):
    """绘图函数，添加边缘直方图和更好的样式"""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # 创建图形和网格布局
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 3)
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)

    # 主散点图
    ax_main.scatter(true_ddg, predictions, alpha=0.6, color='salmon', s=50)

    # 添加对角线
    min_val = min(min(true_ddg), min(predictions))
    max_val = max(max(true_ddg), max(predictions))
    margin = (max_val - min_val) * 0.1
    plot_range = [min_val - margin, max_val + margin]
    ax_main.plot(plot_range, plot_range, '--', color='red', alpha=0.8)

    # 设置轴标签和范围
    ax_main.set_xlabel('Experimental ΔΔG(kcal/mol)')
    ax_main.set_ylabel('Predicted ΔΔG(kcal/mol)')
    ax_main.grid(True, linestyle='--', alpha=0.7)

    # 添加边缘直方图
    bins = np.linspace(plot_range[0], plot_range[1], 30)

    # 顶部直方图 - 使用单一颜色
    ax_top.hist(true_ddg, bins=bins, density=True, alpha=0.7, color='#7CCD7C')
    ax_top.set_xticks([])

    # 右侧直方图 - 使用单一颜色
    ax_right.hist(predictions, bins=bins, density=True, orientation='horizontal',
                  alpha=0.7, color='#7CCD7C')
    ax_right.set_yticks([])

    # 添加指标文本
    if metrics:
        metrics_text = f"PCC={metrics['PCC']:.2f}, RMSE={metrics['RMSE']:.2f}"
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))
        metrics_text = f"PCC={pcc:.2f}, RMSE={rmse:.2f}"

    ax_top.text(0.02, 0.98, metrics_text,
                transform=ax_top.transAxes,
                verticalalignment='top',
                fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_results_plus_good(true_ddg, predictions, save_path, metrics=None):
    """
    Enhanced plotting function with seamless layout and extended grid lines
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    # Create figure
    fig = plt.figure(figsize=(10, 10))

    # Create layout with appropriate spacing
    gs = plt.GridSpec(3, 3, width_ratios=[4, 4, 1], height_ratios=[1, 4, 4],
                      hspace=0.05, wspace=0.05)

    # Create axes
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Calculate data range
    all_values = np.concatenate([true_ddg, predictions])
    min_val = np.floor(min(all_values))
    max_val = np.ceil(max(all_values))
    margin = 0.5
    plot_range = [min_val - margin, max_val + margin]

    # Create grid lines
    grid_interval = 1.0
    grid_lines = np.arange(min_val - margin, max_val + margin + grid_interval, grid_interval)

    # Plot extended grid lines
    for line in grid_lines:
        # Main plot grid lines
        ax_main.axhline(y=line, color='gray', linestyle='-', alpha=0.2, zorder=1)
        ax_main.axvline(x=line, color='gray', linestyle='-', alpha=0.2, zorder=1)
        # Top plot vertical grid lines
        ax_top.axvline(x=line, color='gray', linestyle='-', alpha=0.2, zorder=1)
        # Right plot horizontal grid lines
        ax_right.axhline(y=line, color='gray', linestyle='-', alpha=0.2, zorder=1)

    # Plot scatter points
    ax_main.scatter(true_ddg, predictions, alpha=0.6, color='salmon', s=50, zorder=2)

    # Plot diagonal line
    ax_main.plot(plot_range, plot_range, '--', color='red', alpha=0.8, linewidth=1.5, zorder=3)

    # Create histogram bins with gaps
    bins = np.linspace(plot_range[0], plot_range[1], 25)

    # Plot histograms with reduced width (rwidth) for gaps
    ax_top.hist(true_ddg, bins=bins, density=True, alpha=0.6, color='salmon',
                rwidth=0.8, zorder=2)
    ax_right.hist(predictions, bins=bins, density=True, orientation='horizontal',
                  alpha=0.6, color='salmon', rwidth=0.8, zorder=2)

    # Add density curves
    kde_points = np.linspace(plot_range[0], plot_range[1], 100)

    # Top density curve
    kde_x = gaussian_kde(true_ddg)
    ax_top.plot(kde_points, kde_x(kde_points), color='red', linewidth=2, zorder=3)

    # Right density curve
    kde_y = gaussian_kde(predictions)
    ax_right.plot(kde_y(kde_points), kde_points, color='red', linewidth=2, zorder=3)

    # Set axis limits
    ax_main.set_xlim(plot_range)
    ax_main.set_ylim(plot_range)

    # Configure spines and ticks
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    # Remove specific ticks and labels
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    ax_right.set_xticklabels([])  # Ensure no x-axis labels on right plot
    plt.setp(ax_top.get_xticklabels(), visible=False)  # Hide x-axis labels on top plot
    plt.setp(ax_right.get_yticklabels(), visible=False)  # Hide y-axis labels on right plot

    # Add metrics text
    if metrics:
        metrics_text = f"PCC={metrics['PCC']:.2f}, RMSE={metrics['RMSE']:.2f}"
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))
        metrics_text = f"PCC={pcc:.2f}, RMSE={rmse:.2f}"

    ax_top.text(0.02, 0.98, metrics_text,
                transform=ax_top.transAxes,
                verticalalignment='top',
                fontsize=12)

    # Add labels
    ax_main.set_xlabel('Experimental ΔΔG (kcal/mol)')
    ax_main.set_ylabel('Predicted ΔΔG (kcal/mol)')

    # Make sure the main plot's x-axis labels are visible
    ax_main.xaxis.set_tick_params(labelbottom=True)

    # Adjust layout and save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_results_plus(true_ddg, predictions, save_path, metrics=None):
    """
    Enhanced plotting function combining the best features of both versions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    # Set clean style with white background and grid
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.grid': True,
        'grid.color': 'gray',
        'grid.linestyle': '-',
        'grid.alpha': 0.2,
        'font.size': 10
    })

    # Create figure and layout
    fig = plt.figure(figsize=(10, 10))
    gs = plt.GridSpec(3, 3, width_ratios=[4, 4, 1], height_ratios=[1, 4, 4],
                      hspace=0.05, wspace=0.05)

    # Create axes
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Calculate data range
    all_values = np.concatenate([true_ddg, predictions])
    min_val = np.floor(min(all_values))
    max_val = np.ceil(max(all_values))
    margin = 0.5
    plot_range = [min_val - margin, max_val + margin]

    # Plot scatter points with improved style
    ax_main.scatter(true_ddg, predictions, alpha=0.5, color='salmon', s=40, zorder=3)

    # Plot diagonal line with refined style
    ax_main.plot(plot_range, plot_range, '--', color='red', alpha=0.6, linewidth=1.5, zorder=2)

    # Create histogram bins with gaps
    bins = np.linspace(plot_range[0], plot_range[1], 25)

    # Plot histograms with gaps (using rwidth)
    ax_top.hist(true_ddg, bins=bins, density=True, alpha=0.4, color='salmon',
                rwidth=0.8, zorder=2, edgecolor='none')
    ax_right.hist(predictions, bins=bins, density=True, orientation='horizontal',
                  alpha=0.4, color='salmon', rwidth=0.8, zorder=2, edgecolor='none')

    # Add density curves with refined style
    kde_points = np.linspace(plot_range[0], plot_range[1], 200)

    # Top density curve
    kde_x = gaussian_kde(true_ddg)
    ax_top.plot(kde_points, kde_x(kde_points), color='salmon', linewidth=1.5, zorder=3)

    # Right density curve
    kde_y = gaussian_kde(predictions)
    ax_right.plot(kde_y(kde_points), kde_points, color='salmon', linewidth=1.5, zorder=3)

    # Set axis limits
    ax_main.set_xlim(plot_range)
    ax_main.set_ylim(plot_range)

    # Configure spines
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    # Remove specific ticks and labels
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Enable grid for main plot
    ax_main.grid(True, linestyle='-', alpha=0.2)
    ax_main.set_axisbelow(True)  # Ensure grid is below data points

    # Add metrics text with refined position and style
    if metrics:
        metrics_text = f"PCC={metrics['PCC']:.2f}, RMSE={metrics['RMSE']:.2f}"
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))
        metrics_text = f"PCC={pcc:.2f}, RMSE={rmse:.2f}"

    ax_top.text(0.02, 0.85, metrics_text,
                transform=ax_top.transAxes,
                verticalalignment='top',
                fontsize=12,
                color='black')

    # Add labels with refined style
    ax_main.set_xlabel('Experimental ΔΔG (kcal/mol)', fontsize=11)
    ax_main.set_ylabel('Predicted ΔΔG (kcal/mol)', fontsize=11)

    # Fine-tune tick parameters
    ax_main.tick_params(direction='out', length=4, width=1)
    ax_main.tick_params(which='both', bottom=True, top=False, left=True, right=False)

    # Ensure main plot's x-axis labels are visible
    ax_main.xaxis.set_tick_params(labelbottom=True)

    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_results(true_ddg, predictions, save_path):
    """绘制预测结果对比图"""
    plt.figure(figsize=(8, 8))
    plt.scatter(true_ddg, predictions, alpha=0.5)
    plt.plot([min(true_ddg), max(true_ddg)], [min(true_ddg), max(true_ddg)], 'r--')
    plt.xlabel('Experimental ΔΔG (kcal/mol)')
    plt.ylabel('Predicted ΔΔG (kcal/mol)')
    plt.title('Prediction vs Experiment')
    plt.savefig(save_path)
    plt.close()


def calculate_metrics(true_ddg, predictions):
    """计算各种评估指标"""
    return {
        'PCC': pearsonr(true_ddg, predictions)[0],
        'RMSE': np.sqrt(mean_squared_error(true_ddg, predictions)),
        'MAE': mean_absolute_error(true_ddg, predictions),
        'R2': np.corrcoef(true_ddg, predictions)[0, 1] ** 2,
        'Mean Error': np.mean(predictions - true_ddg),
        'Std Error': np.std(predictions - true_ddg)
    }


def print_header(text, width=50):
    """打印美观的标题"""
    print("\n" + "=" * width)
    print(f"{text:^{width}}")
    print("=" * width + "\n")


def print_section(text, width=50):
    """打印小节标题"""
    print(f"\n{'-' * 5} {text} {'-' * (width - len(text) - 7)}")


def format_percentage(value, total):
    """格式化百分比"""
    return f"{value} ({value / total * 100:.1f}%)"


def print_table(headers, rows, column_widths=None):
    """简单的表格打印函数，替代tabulate"""
    if column_widths is None:
        # 计算每列的最大宽度
        widths = []
        for i in range(len(headers)):
            col_items = [str(row[i]) for row in rows] + [str(headers[i])]
            widths.append(max(len(item) for item in col_items) + 2)
    else:
        widths = column_widths

    # 打印表头
    header = " ".join(f"{str(h):<{w}}" for h, w in zip(headers, widths))
    print(header)
    print("-" * sum(widths))

    # 打印数据行
    for row in rows:
        print(" ".join(f"{str(item):<{w}}" for item, w in zip(row, widths)))


def log_results(metrics, predictions, true_ddg, mutant_names, model_path, test_data_path):
    """使用print重构的结果输出函数"""
    print(f"\nPrediction completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print_header("PREDICTION RESULTS")

    # 1. 模型和数据信息
    print_section("Model Information")
    info_rows = [
        ["Model Path", model_path],
        ["Test Dataset", test_data_path],
        ["Number of Samples", len(predictions)]
    ]
    print_table(["Item", "Value"], info_rows)

    # 2. 性能指标
    print_section("Performance Metrics")
    metrics_rows = [[k, f"{v:.4f}"] for k, v in metrics.items()]
    print_table(["Metric", "Value"], metrics_rows)

    # 3. 创建结果DataFrame
    results_df = pd.DataFrame({
        'Mutant': mutant_names,
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    # 4. 预测统计
    print_section("Prediction Statistics")
    total = len(predictions)
    within_1 = (results_df['Absolute_Error'] <= 1.0).sum()
    within_2 = (results_df['Absolute_Error'] <= 2.0).sum()

    stats_rows = [
        ["Total Mutations", total],
        ["Within 1 kcal/mol", format_percentage(within_1, total)],
        ["Within 2 kcal/mol", format_percentage(within_2, total)]
    ]
    print_table(["Statistic", "Value"], stats_rows)

    # 5. 错误分布分析
    print_section("Error Distribution")
    error_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, float('inf'))]
    error_rows = []
    for low, high in error_ranges:
        mask = (results_df['Absolute_Error'] > low) & (results_df['Absolute_Error'] <= high)
        count = mask.sum()
        error_rows.append([
            f"{low}-{high if high != float('inf') else '∞'} kcal/mol",
            format_percentage(count, total)
        ])
    print_table(["Range", "Count"], error_rows)

    # 6. DDG范围分析
    print_section("DDG Range Analysis")
    ddg_ranges = [(-float('inf'), -3), (-3, -1), (-1, 1), (1, 3), (3, float('inf'))]
    ddg_rows = []
    for low, high in ddg_ranges:
        mask = (results_df['True_DDG'] > low) & (results_df['True_DDG'] <= high)
        if mask.any():
            subset = results_df[mask]
            mae = subset['Absolute_Error'].mean()
            count = len(subset)
            range_str = f"{low:>3.0f} to {high:<3.0f}" if low != float('-inf') else f"< {high:<3.0f}"
            ddg_rows.append([range_str, format_percentage(count, total), f"{mae:.2f}"])
    print_table(["DDG Range", "Count", "MAE (kcal/mol)"], ddg_rows)

    # 7. 最大/最小错误案例
    print_section("Largest Prediction Errors")
    largest_errors = results_df.nlargest(5, 'Absolute_Error')[
        ['Mutant', 'True_DDG', 'Predicted_DDG', 'Absolute_Error']
    ]
    print(largest_errors.to_string(index=False))

    print_section("Smallest Prediction Errors")
    smallest_errors = results_df.nsmallest(5, 'Absolute_Error')[
        ['Mutant', 'True_DDG', 'Predicted_DDG', 'Absolute_Error']
    ]
    print(smallest_errors.to_string(index=False))

    print("\nResults have been saved to:")
    print("- enhanced_predictions_detailed.csv")
    print("- prediction_enhanced_analysis.png")

    return results_df


# 保持其他函数不变
def save_results(predictions, true_ddg, metrics, filename='enhanced_predictions.csv'):
    """保存预测结果和指标"""
    results = pd.DataFrame({
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    for metric, value in metrics.items():
        results.attrs[metric] = value

    results.to_csv(filename, index=False)
    logging.info(f"Results saved to {filename}")


def save_detailed_results(predictions, true_ddg, mutant_names, metrics, filename):
    """保存详细的预测结果并输出完整统计信息"""
    results = pd.DataFrame({
        'Mutant': mutant_names,
        'True_DDG': true_ddg,
        'Predicted_DDG': predictions,
        'Absolute_Error': np.abs(predictions - true_ddg)
    })

    for metric, value in metrics.items():
        results.attrs[metric] = value

    results = results.sort_values('Absolute_Error', ascending=False)
    results.to_csv(filename, index=False)

    logging.info(f"Detailed results saved to {filename}")
    return results