import os
import torch
from torch_geometric.data import Data, Batch
import pickle
import numpy as np
import logging


class EnhancedDataLoader:
    def __init__(self, data_path, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 配置日志
        logging.basicConfig(level=logging.INFO)

        # 加载数据
        if isinstance(data_path, str):
            with open(data_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = data_path

        # 验证数据格式
        self._validate_data_format()

        # 创建索引列表
        self.indices = list(range(len(self.dataset)))

    def _validate_data_format(self):
        """验证数据格式"""
        if not self.dataset:
            # raise ValueError("Dataset is empty")
            logging.warning("Dataset is empty")

        first_item = self.dataset[0]
        required_keys = {'wild_type', 'mutant', 'ddg'}
        if not all(key in first_item for key in required_keys):
            raise ValueError(f"Missing required keys in data: {required_keys - set(first_item.keys())}")

        # 打印数据结构以便调试
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
        """创建批处理，支持mutation_pos属性"""
        try:
            data_objects = []
            for i, data in enumerate(data_list):
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
                mutation_pos = self._get_tensor_data(data, 'mutation_pos')

                # 创建批处理索引
                batch = torch.zeros(x.size(0), dtype=torch.long)

                # 创建Data对象，确保包含mutation_pos
                data_obj = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=batch,
                    mutation_pos=mutation_pos  # 确保包含mutation_pos
                )
                data_objects.append(data_obj)

            # 使用Batch.from_data_list创建批处理
            batch = Batch.from_data_list(data_objects)

            # 确保mutation_pos正确传递
            batch.mutation_pos = torch.cat([d.mutation_pos for d in data_objects])

            return batch

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


def prepare_enhanced_data(data_path, batch_size=32, val_split=0.1):
    """准备训练和验证数据加载器"""
    logging.info(f"Loading data from: {data_path}")

    try:
        # 创建数据加载器
        dataset = []
        if isinstance(data_path, str):
            with open(data_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = data_path

        # 计算划分大小
        total_size = len(dataset)
        val_size = max(1, int(total_size * val_split))
        indices = np.random.permutation(total_size)

        # 划分数据集
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        # 创建训练集
        train_dataset = [dataset[i] for i in train_indices]
        train_loader = EnhancedDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 创建验证集
        val_dataset = [dataset[i] for i in val_indices]
        val_loader = EnhancedDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logging.info(f"Successfully created data loaders:")
        logging.info(f"Training samples: {len(train_dataset)}")
        logging.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader

    except Exception as e:
        logging.error(f"Error preparing data: {str(e)}")
        raise