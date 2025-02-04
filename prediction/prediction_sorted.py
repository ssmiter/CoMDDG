"""prediction_com_sorted.py - 基于节点度排序的Bidirectional Mamba模型预测脚本"""
import pandas as pd
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import NUM_LAYERS, USE_SUBGRAPHS, NUM_HOPS
from utils import log_results, plot_results_plus, plot_results_plus_good
from model.CoM_Bidirectional_sorted_cognn import CoGNN_GraphMambaSorted
from model.utils.loader.data_loader_add_mut_v2 import prepare_test_data


def predict(model_path, test_data_path, batch_size=16):
    """
    使用基于节点度排序的Bidirectional Mamba模型进行预测
    
    Args:
        model_path (str): 模型权重文件路径
        test_data_path (str): 测试数据文件路径
        batch_size (int): 批处理大小
    """
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载测试数据
    test_loader = prepare_test_data(
        data_path=test_data_path,
        batch_size=batch_size,
    )

    # 获取特征维度
    sample_wild, sample_mutant, _ = next(iter(test_loader))
    node_features = sample_wild.x.shape[1]

    # 初始化模型
    model = CoGNN_GraphMambaSorted(
        in_channels=node_features,
        hidden_channels=64,
        out_channels=1,
        # gmb_args={ # case1
        #     'd_model': 64,  # Match hidden_channels
        #     'd_state': 16,
        #     'd_conv': 4,
        #     'expand': 2,
        #     'use_checkpointing': True
        # },

        gmb_args={  # case2
            'd_model': 64,  # Match hidden_channels
            'd_state': 16,
            'd_conv': 2,
            'expand': 1,
            # 'use_checkpointing': True
        },
        num_layers=NUM_LAYERS
    ).to(device)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded model from epoch {checkpoint['epoch']} with:")
    print(f"- Validation loss: {checkpoint['loss']:.4f}")
    print(f"- Validation MAE: {checkpoint['val_mae']:.4f}")
    print(f"- Validation MSE: {checkpoint['val_mse']:.4f}")
    if 'val_pcc' in checkpoint:
        print(f"- Validation PCC: {checkpoint['val_pcc']:.4f}")

    # 预测
    model.eval()
    predictions = []
    true_ddg = []
    mutant_names = []

    print("\nStarting predictions...")
    with torch.no_grad():
        for batch_idx, (wild_data, mutant_data, ddg) in enumerate(test_loader):
            wild_data = wild_data.to(device)
            mutant_data = mutant_data.to(device)

            # 进行预测
            output = model(wild_data, mutant_data)

            predictions.extend(output.cpu().numpy())
            true_ddg.extend(ddg.numpy())

            if hasattr(wild_data, 'mutation_name'):
                mutant_names.extend(wild_data.mutation_name)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches")

    # 转换为numpy数组
    predictions = np.array(predictions)
    true_ddg = np.array(true_ddg)

    # 计算评估指标
    metrics = {
        'PCC': pearsonr(true_ddg, predictions)[0],
        'RMSE': np.sqrt(mean_squared_error(true_ddg, predictions)),
        'MAE': mean_absolute_error(true_ddg, predictions)
    }

    print("\nEvaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # 如果没有突变名称，生成默认名称
    if not mutant_names:
        mutant_names = [f"Mutation_{i + 1}" for i in range(len(predictions))]

    # 记录结果
    results_df = log_results(
        metrics, predictions, true_ddg, mutant_names,
        model_path, test_data_path
    )

    # 绘制结果图
    plot_results_plus(
        true_ddg,
        predictions,
        f'./result/degree_sorted_mamba_predictions_on_S{dataset}{end}.png',
        metrics
    )

    # 保存详细结果到CSV
    results_df.to_csv(
        f'./result/degree_sorted_mamba_predictions_on_{dataset}{end}.csv',
        index=False
    )
    print(f"\nResults have been saved to degree_sorted_mamba_predictions_on_{dataset}{end}.csv")


if __name__ == "__main__":
    # 配置参数
    # dataset = "ssym"  # 数据集名称
    # dataset = "s605"  # 数据集名称
    dataset = "s250"  # 数据集名称
    end = "_cognn"  # 文件名后缀
    train_dataset = '2648'

    # model_path = f'../training/CoM_Bidirectional_Sorted_S{train_dataset}_struct.pth'
    # model_path = f'../training/CoM_Bidirectional_Sorted_S{train_dataset}_struct_cognn_4layers(ssym).pth'
    # model_path = f'../training/CoM_Bidirectional_Sorted_S2648_struct_cognn_4layers(best_ssym).pth'
    model_path = f'../training/CoM_Bidirectional_Sorted_S2648_struct_cognn_4layers_attempt_1.pth'
    test_data_path = f'../dataset_process/pkl/data_{dataset}_add_mutpos_enhanced_struct.pkl'

    # 运行预测
    predict(model_path, test_data_path)
