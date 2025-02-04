import os
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.CoM_Bidirectional_sorted_cognn import CoGNN_GraphMambaSorted
from model.CoM_Bidirectional_sorted_cognn_Mamba_only import MambaOnly
from model.CoM_Bidirectional_sorted_cognn_only import CoGNNOnly
from model.utils.loader.data_loader_add_mut_v2 import prepare_test_data
from utils import log_results, plot_results_plus
from config import NUM_LAYERS


def predict_all_attempts(base_model_path, test_datasets, batch_size=16):
    """
    预测所有尝试模型的结果

    Args:
        base_model_path (str): 基础模型路径（不包含_attempt_x.pth）
        test_datasets (list): 测试数据集列表，每个元素是(dataset_name, data_path)的元组
        batch_size (int): 批处理大小
    """
    # 获取所有模型文件
    model_dir = os.path.dirname(base_model_path)
    base_name = os.path.basename(base_model_path).replace('.pth', '')
    model_files = [f for f in os.listdir(model_dir) if f.startswith(base_name) and 'attempt' in f]

    # 创建结果存储DataFrame
    summary_results = []

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for model_file in sorted(model_files, key=lambda x: int(x.split('attempt_')[1].split('.')[0])):
        model_path = os.path.join(model_dir, model_file)
        attempt_num = int(model_file.split('attempt_')[1].split('.')[0])
        print(f"\nProcessing model: {model_file} (Attempt {attempt_num})")

        # 对每个测试数据集进行预测
        for dataset_name, test_data_path in test_datasets:
            print(f"\nTesting on dataset: {dataset_name}")

            # 加载测试数据
            test_loader = prepare_test_data(
                data_path=test_data_path,
                batch_size=batch_size
            )

            # 获取特征维度并初始化模型
            sample_wild, sample_mutant, _ = next(iter(test_loader))
            node_features = sample_wild.x.shape[1]

            model = CoGNN_GraphMambaSorted(  # model = CoGNN_GraphMambaSorted or MambaOnly or CoGNNOnly
                in_channels=node_features,
                hidden_channels=64,
                out_channels=1,
                gmb_args={
                    'd_model': 64,
                    'd_state': 16,
                    'd_conv': 2,
                    'expand': 1,
                },
                num_layers=NUM_LAYERS
            ).to(device)

            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 预测
            model.eval()
            predictions = []
            true_ddg = []
            mutant_names = []

            print("Making predictions...")
            with torch.no_grad():
                for batch_idx, (wild_data, mutant_data, ddg) in enumerate(test_loader):
                    wild_data = wild_data.to(device)
                    mutant_data = mutant_data.to(device)

                    output = model(wild_data, mutant_data)
                    predictions.extend(output.cpu().numpy())
                    true_ddg.extend(ddg.numpy())

                    if hasattr(wild_data, 'mutation_name'):
                        mutant_names.extend(wild_data.mutation_name)

            # 计算评估指标
            predictions = np.array(predictions)
            true_ddg = np.array(true_ddg)

            pcc = pearsonr(true_ddg, predictions)[0]
            rmse = np.sqrt(mean_squared_error(true_ddg, predictions))
            mae = mean_absolute_error(true_ddg, predictions)

            print(f"Results for attempt {attempt_num} on {dataset_name}:")
            print(f"PCC: {pcc:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")

            # 保存结果
            results_dir = os.path.join('./result', f'attempt_{attempt_num}')
            os.makedirs(results_dir, exist_ok=True)

            # 如果没有突变名称，生成默认名称
            if not mutant_names:
                mutant_names = [f"Mutation_{i + 1}" for i in range(len(predictions))]

            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'Mutation': mutant_names,
                'Predicted_DDG': predictions.flatten(),
                'True_DDG': true_ddg
            })

            # 保存详细结果
            results_df.to_csv(
                os.path.join(results_dir, f'predictions_{dataset_name}.csv'),
                index=False
            )

            # 绘制结果图
            plot_results_plus(
                true_ddg,
                predictions,
                os.path.join(results_dir, f'predictions_{dataset_name}.png'),
                {'PCC': pcc, 'RMSE': rmse, 'MAE': mae}
            )

            # 添加到总结果
            summary_results.append({
                'Attempt': attempt_num,
                'Dataset': dataset_name,
                'PCC': pcc,
                'RMSE': rmse,
                'MAE': mae,
                'Model_Path': model_file,
                'Original_Val_PCC': checkpoint.get('val_pcc', None),
                'Original_Val_MAE': checkpoint.get('val_mae', None),
                'Original_Val_MSE': checkpoint.get('val_mse', None)
            })

    # 创建和保存总结果表格
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('./result/all_attempts_summary.csv', index=False)

    # 打印最佳结果
    print("\nBest results for each dataset:")
    for dataset_name, _ in test_datasets:
        dataset_results = summary_df[summary_df['Dataset'] == dataset_name]
        best_attempt = dataset_results.loc[dataset_results['PCC'].idxmax()]
        print(f"\n{dataset_name}:")
        print(f"Best attempt: {best_attempt['Attempt']}")
        print(f"PCC: {best_attempt['PCC']:.4f}")
        print(f"RMSE: {best_attempt['RMSE']:.4f}")
        print(f"MAE: {best_attempt['MAE']:.4f}")

    return summary_df


if __name__ == "__main__":
    # 配置基础模型路径
    base_model_path = '../training/CoM_Bidirectional_Sorted_S2648_struct_cognn_4layers.pth'
    # base_model_path = '../training/CoM_Bidirectional_Sorted_S2648_struct_cognn_Mamba_only.pth'
    # base_model_path = '../training/CoM_Bidirectional_Sorted_S2648_struct_cognn_only.pth'

    # 配置测试数据集
    test_datasets = [
        # ('ssym', '../dataset_process/pkl/data_ssym_add_mutpos_enhanced_struct.pkl'),
        # ('s605', '../dataset_process/pkl/data_s605_add_mutpos_enhanced_struct.pkl'),
        # ('s350', '../dataset_process/pkl/data_s350_add_mutpos_enhanced_struct.pkl'),
        ('s250', '../dataset_process/pkl/data_s250_add_mutpos_enhanced_struct.pkl'),
        # ('s1925', '../dataset_process/pkl/data_s1925_add_mutpos_enhanced_struct.pkl'),
        # 可以添加更多数据集
    ]

    # 运行批量预测
    summary_results = predict_all_attempts(base_model_path, test_datasets)
