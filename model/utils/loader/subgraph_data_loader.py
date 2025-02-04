import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pickle
from config import BATCH_SIZE


def load_subgraph_dataset(file_path):
    """加载预处理后的子图数据集"""
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    data_list = []
    for item in dataset:
        # 创建野生型子图的Data对象
        wild_subgraph = item['wild_type_subgraph']
        wild_data = Data(
            x=torch.tensor(wild_subgraph['node_features'], dtype=torch.float),
            edge_index=torch.tensor(wild_subgraph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(wild_subgraph['edge_features'], dtype=torch.float),
            mutation_idx=torch.tensor(wild_subgraph['mutation_idx'], dtype=torch.long),
            batch=torch.zeros(wild_subgraph['node_features'].shape[0], dtype=torch.long)
        )

        # 创建突变型子图的Data对象
        mutant_subgraph = item['mutant_subgraph']
        mutant_data = Data(
            x=torch.tensor(mutant_subgraph['node_features'], dtype=torch.float),
            edge_index=torch.tensor(mutant_subgraph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(mutant_subgraph['edge_features'], dtype=torch.float),
            mutation_idx=torch.tensor(mutant_subgraph['mutation_idx'], dtype=torch.long),
            batch=torch.zeros(mutant_subgraph['node_features'].shape[0], dtype=torch.long)
        )

        data_list.append((wild_data, mutant_data, torch.tensor(item['ddg'], dtype=torch.float)))

    return data_list


def prepare_subgraph_data(file_path):
    """准备训练和验证数据加载器"""
    dataset = load_subgraph_dataset(file_path)

    # 创建反向突变数据
    additional_data = []
    for wild_data, mutant_data, ddg in dataset:
        additional_data.append((mutant_data, wild_data, ddg * -1))
    datasets = dataset + additional_data

    # 划分训练集和验证集
    train_data, val_data = train_test_split(datasets, test_size=0.2, random_state=42)

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader