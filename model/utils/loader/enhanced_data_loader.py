import os
import torch
from torch_geometric.data import Data, Batch
import pickle
import numpy as np
import logging
from typing import List, Tuple, Union

from torch_geometric.loader import DataLoader


class EnhancedDataLoader:
    def __init__(self, data_path, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        logging.basicConfig(level=logging.INFO)

        # 加载数据
        if isinstance(data_path, str):
            with open(data_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = data_path

        self._validate_data_format()
        self.indices = list(range(len(self.dataset)))

    def _validate_data_format(self):
        if not self.dataset:
            raise ValueError("Dataset is empty")

        first_item = self.dataset[0]
        required_keys = {'wild_type', 'mutant', 'ddg'}
        if not all(key in first_item for key in required_keys):
            raise ValueError(f"Missing required keys in data: {required_keys - set(first_item.keys())}")

        logging.info(f"Data structure:")
        logging.info(f"Number of samples: {len(self.dataset)}")
        logging.info(f"First item keys: {first_item.keys()}")
        logging.info(f"Wild type data structure: {first_item['wild_type'].keys()}")
        logging.info(f"Mutant data structure: {first_item['mutant'].keys()}")

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            try:
                batch_indices = self.indices[i:i + self.batch_size]
                batch_data = [self.dataset[idx] for idx in batch_indices]

                wild_batch = self._create_batch([item['wild_type'] for item in batch_data])
                mutant_batch = self._create_batch([item['mutant'] for item in batch_data])
                ddg = torch.tensor([item['ddg'] for item in batch_data], dtype=torch.float)

                yield wild_batch, mutant_batch, ddg

            except Exception as e:
                logging.error(f"Error processing batch at index {i}: {str(e)}")
                continue

    def _create_batch(self, data_list):
        """创建批处理，支持多种数据格式"""
        try:
            data_objects = []
            for data in data_list:
                # 处理节点特征
                x = self._get_tensor_data(data, 'x')
                if x is None:
                    x = self._get_tensor_data(data, 'node_features')

                # 处理边索引
                edge_index = self._get_tensor_data(data, 'edge_index')

                # 处理边特征
                edge_attr = self._get_tensor_data(data, 'edge_attr')
                if edge_attr is None:
                    edge_attr = self._get_tensor_data(data, 'edge_features')

                # 处理突变位置
                mutation_pos = data.get('mutation_pos', None)
                if mutation_pos is not None:
                    mutation_pos = self._get_tensor_data(data, 'mutation_pos')

                # 确保所有必要的数据都存在
                if any(v is None for v in [x, edge_index, edge_attr]):
                    raise ValueError("Missing required data fields")

                # 创建批处理索引
                batch = torch.zeros(x.size(0), dtype=torch.long)

                # 创建Data对象
                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch
                )

                if mutation_pos is not None:
                    data_obj.mutation_pos = mutation_pos

                data_objects.append(data_obj)

            return Batch.from_data_list(data_objects)

        except Exception as e:
            logging.error(f"Error creating batch: {str(e)}")
            raise

    def _get_tensor_data(self, data, key):
        """安全地获取并转换张量数据"""
        try:
            if key not in data:
                return None

            value = data[key]
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            else:
                return torch.tensor(value)
        except Exception as e:
            logging.error(f"Error converting data for key {key}: {str(e)}")
            return None


def create_reverse_mutations(dataset: List[dict]) -> List[dict]:
    """创建反向突变数据"""
    reverse_mutations = []
    for item in dataset:
        # 创建反向突变数据
        reverse_item = {
            'wild_type': item['mutant'],  # 交换野生型和突变型
            'mutant': item['wild_type'],
            'ddg': -item['ddg'],  # DDG取反
            'wild_type_name': item['mutant_name'],  # 交换名称
            'mutant_name': item['wild_type_name']
        }
        reverse_mutations.append(reverse_item)
    return reverse_mutations


def prepare_enhanced_data(data_path: Union[str, List], batch_size: int = 32, val_split: float = 0.1) -> Tuple[
    EnhancedDataLoader, EnhancedDataLoader]:
    """准备训练和验证数据加载器，包含反向突变"""
    logging.info(f"Loading and preparing data...")

    try:
        # 加载原始数据集
        if isinstance(data_path, str):
            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = data_path

        # 创建反向突变数据
        reverse_mutations = create_reverse_mutations(dataset)

        # 合并原始数据集和反向突变数据集
        full_dataset = dataset + reverse_mutations

        logging.info(f"Original dataset size: {len(dataset)}")
        logging.info(f"Full dataset size (with reverse mutations): {len(full_dataset)}")

        # 随机打乱数据
        total_size = len(full_dataset)
        val_size = max(1, int(total_size * val_split))
        indices = np.random.permutation(total_size)

        # 划分训练集和验证集
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # 创建训练集和验证集
        train_dataset = [full_dataset[i] for i in train_indices]
        val_dataset = [full_dataset[i] for i in val_indices]

        # 创建数据加载器
        train_loader = EnhancedDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = EnhancedDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logging.info(f"Data preparation completed:")
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise


def prepare_test_data(data_path, batch_size=16):
    """专用于测试的数据加载函数，不创建反向突变"""
    logging.info(f"Loading test data from: {data_path}")

    try:
        # 加载原始数据
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        # 创建测试数据列表
        test_data = []
        for item in dataset:
            try:
                # 构建野生型数据
                wild_data = {
                    'x': torch.tensor(item['wild_type']['x'], dtype=torch.float),
                    'edge_index': torch.tensor(item['wild_type']['edge_index'], dtype=torch.long),
                    'edge_attr': torch.tensor(item['wild_type']['edge_attr'], dtype=torch.float),
                    'batch': torch.zeros(len(item['wild_type']['x']), dtype=torch.long)
                }

                # 构建突变型数据
                mutant_data = {
                    'x': torch.tensor(item['mutant']['x'], dtype=torch.float),
                    'edge_index': torch.tensor(item['mutant']['edge_index'], dtype=torch.long),
                    'edge_attr': torch.tensor(item['mutant']['edge_attr'], dtype=torch.float),
                    'batch': torch.zeros(len(item['mutant']['x']), dtype=torch.long)
                }

                # 保存原始突变信息
                test_data.append((
                    wild_data,
                    mutant_data,
                    torch.tensor(item['ddg'], dtype=torch.float),
                    item.get('mutant_name', 'unknown')  # 添加突变名称，如果存在的话
                ))

            except Exception as e:
                logging.warning(f"Error processing item: {e}")
                continue

        logging.info(f"Successfully loaded {len(test_data)} test samples")

        # 创建DataLoader
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False  # 测试时不打乱顺序
        )

        return test_loader

    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise