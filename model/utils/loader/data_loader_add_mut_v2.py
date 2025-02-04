"""支持可选的子图提取功能"""
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from config import BATCH_SIZE, NUM_HOPS, SEED, USE_SUBGRAPHS


class EnhancedDataLoader:
    """增强版数据加载器，支持可选的子图提取功能"""
    def __init__(self, data_path, batch_size=BATCH_SIZE, val_split=0.1, num_hops=NUM_HOPS, use_subgraphs=USE_SUBGRAPHS):
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.use_subgraphs = use_subgraphs
        self.data_list = self.load_and_process_data(data_path)
        self.train_loader, self.val_loader = self.prepare_data_loaders(val_split)

    def extract_subgraph(self, node_features, edge_index, edge_features, mutation_pos):
        """提取以突变位点为中心的子图"""
        if not self.use_subgraphs:
            # 如果不使用子图，直接返回原始图的数据
            return (node_features, edge_index, edge_features, mutation_pos)

        try:
            # 确保数据类型正确
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            mutation_pos = torch.tensor([mutation_pos], dtype=torch.long)

            # 提取k-hop子图
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=mutation_pos,
                num_hops=self.num_hops,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=len(node_features)
            )

            # 获取子图特征
            sub_nodes = node_features[subset.numpy()]
            sub_edges = sub_edge_index.numpy()
            sub_edge_features = edge_features[edge_mask.numpy()]
            new_mutation_pos = mapping[0].item()

            return sub_nodes, sub_edges, sub_edge_features, new_mutation_pos
            
        except Exception as e:
            print(f"Error extracting subgraph: {str(e)}")
            return None

    def load_and_process_data(self, data_path):
        """加载并处理数据集"""
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        processed_data = []
        for item in dataset:
            try:
                # 处理野生型数据
                wild_subgraph = self.extract_subgraph(
                    item['wild_type']['node_features'],
                    item['wild_type']['edge_index'],
                    item['wild_type']['edge_features'],
                    item['wild_type']['mutation_pos']
                )
                
                # 处理突变型数据
                mutant_subgraph = self.extract_subgraph(
                    item['mutant']['node_features'],
                    item['mutant']['edge_index'],
                    item['mutant']['edge_features'],
                    item['mutant']['mutation_pos']
                )

                if wild_subgraph and mutant_subgraph:
                    w_nodes, w_edges, w_edge_feat, w_mut_idx = wild_subgraph
                    m_nodes, m_edges, m_edge_feat, m_mut_idx = mutant_subgraph

                    # 创建Data对象
                    wild_data = Data(
                        x=torch.tensor(w_nodes, dtype=torch.float),
                        edge_index=torch.tensor(w_edges, dtype=torch.long),
                        edge_attr=torch.tensor(w_edge_feat, dtype=torch.float),
                        mutation_pos=torch.tensor([w_mut_idx], dtype=torch.long),
                        batch=torch.zeros(len(w_nodes), dtype=torch.long)
                    )

                    mutant_data = Data(
                        x=torch.tensor(m_nodes, dtype=torch.float),
                        edge_index=torch.tensor(m_edges, dtype=torch.long),
                        edge_attr=torch.tensor(m_edge_feat, dtype=torch.float),
                        mutation_pos=torch.tensor([m_mut_idx], dtype=torch.long),
                        batch=torch.zeros(len(m_nodes), dtype=torch.long)
                    )

                    processed_data.append((
                        wild_data,
                        mutant_data,
                        torch.tensor(item['ddg'], dtype=torch.float)
                    ))

            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue

        return processed_data

    def prepare_data_loaders(self, val_split):
        """准备训练和验证数据加载器"""
        # 创建反向突变数据
        additional_data = []
        for wild_data, mutant_data, ddg in self.data_list:
            additional_data.append((mutant_data, wild_data, -ddg))
        
        full_dataset = self.data_list + additional_data

        # 划分训练集和验证集
        train_data, val_data = train_test_split(
            full_dataset, 
            test_size=val_split, 
            random_state=SEED
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=self.batch_size, 
            shuffle=False
        )

        return train_loader, val_loader


def prepare_enhanced_data(data_path, batch_size=BATCH_SIZE, val_split=0.1):
    """便捷函数用于创建数据加载器"""
    data_loader = EnhancedDataLoader(
        data_path=data_path,
        batch_size=batch_size,
        val_split=val_split,
    )
    return data_loader.train_loader, data_loader.val_loader


class EnhancedTestDataLoader:
    """专用于测试/预测的数据加载器"""

    def __init__(self, data_path, batch_size=16, num_hops=NUM_HOPS, use_subgraphs=USE_SUBGRAPHS):
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.use_subgraphs = use_subgraphs
        self.data_list = self.load_and_process_data(data_path)
        self.test_loader = self.prepare_test_loader()

    def extract_subgraph(self, node_features, edge_index, edge_features, mutation_pos):
        """提取以突变位点为中心的子图"""
        if not self.use_subgraphs:
            # 如果不使用子图，直接返回原始图的数据
            return (node_features, edge_index, edge_features, mutation_pos)

        try:
            # 确保数据类型正确
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            mutation_pos = torch.tensor([mutation_pos], dtype=torch.long)

            # 提取k-hop子图
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=mutation_pos,
                num_hops=self.num_hops,
                edge_index=edge_index,
                relabel_nodes=True,
                num_nodes=len(node_features)
            )

            # 获取子图特征
            sub_nodes = node_features[subset.numpy()]
            sub_edges = sub_edge_index.numpy()
            sub_edge_features = edge_features[edge_mask.numpy()]
            new_mutation_pos = mapping[0].item()

            return sub_nodes, sub_edges, sub_edge_features, new_mutation_pos

        except Exception as e:
            print(f"Error extracting subgraph: {str(e)}")
            return None

    def load_and_process_data(self, data_path):
        """加载并处理数据集"""
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        processed_data = []
        for item in dataset:
            try:
                # 处理野生型数据
                wild_subgraph = self.extract_subgraph(
                    item['wild_type']['node_features'],
                    item['wild_type']['edge_index'],
                    item['wild_type']['edge_features'],
                    item['wild_type']['mutation_pos']
                )

                # 处理突变型数据
                mutant_subgraph = self.extract_subgraph(
                    item['mutant']['node_features'],
                    item['mutant']['edge_index'],
                    item['mutant']['edge_features'],
                    item['mutant']['mutation_pos']
                )

                if wild_subgraph and mutant_subgraph:
                    w_nodes, w_edges, w_edge_feat, w_mut_idx = wild_subgraph
                    m_nodes, m_edges, m_edge_feat, m_mut_idx = mutant_subgraph

                    # 创建Data对象
                    wild_data = Data(
                        x=torch.tensor(w_nodes, dtype=torch.float),
                        edge_index=torch.tensor(w_edges, dtype=torch.long),
                        edge_attr=torch.tensor(w_edge_feat, dtype=torch.float),
                        mutation_pos=torch.tensor([w_mut_idx], dtype=torch.long),
                        mutation_name=item.get('mutant_name', ''),  # 添加突变名称
                        batch=torch.zeros(len(w_nodes), dtype=torch.long)
                    )

                    mutant_data = Data(
                        x=torch.tensor(m_nodes, dtype=torch.float),
                        edge_index=torch.tensor(m_edges, dtype=torch.long),
                        edge_attr=torch.tensor(m_edge_feat, dtype=torch.float),
                        mutation_pos=torch.tensor([m_mut_idx], dtype=torch.long),
                        mutation_name=item.get('mutant_name', ''),  # 添加突变名称
                        batch=torch.zeros(len(m_nodes), dtype=torch.long)
                    )

                    processed_data.append((
                        wild_data,
                        mutant_data,
                        torch.tensor(item['ddg'], dtype=torch.float)
                    ))

            except Exception as e:
                print(f"Error processing item: {str(e)}")
                continue

        return processed_data

    def prepare_test_loader(self):
        """准备测试数据加载器"""
        return DataLoader(
            self.data_list,
            batch_size=self.batch_size,
            shuffle=False
        )


def prepare_test_data(data_path, batch_size=16):
    """便捷函数用于创建测试数据加载器"""
    data_loader = EnhancedTestDataLoader(
        data_path=data_path,
        batch_size=batch_size,
    )
    return data_loader.test_loader