# config.py
def get_config(node_features):

    gmb_args = {
        'd_model': 64,  # 应与 hidden_channels 匹配
        'd_state': 8,
        'd_conv': 2,
        'expand': 1,
        'use_checkpointing': True
    }

    model_args = {
        'in_channels': node_features,
        'hidden_channels': 64,
        'out_channels': 1,
        'gmb_args': gmb_args,
        'num_layers': 3
    }

    return model_args


# 全局配置: 默认
ENCODING_DIM = 16
BATCH_SIZE = 128        # 可以适当调整batch size，因为每个样本现在使用突变局部子图
LEARNING_RATE = 0.0012   # 可以适当调整学习率
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 200        # 可以适当调整, 建议全图300，子图100
NUM_LAYERS = 4
SEED = 42  # 41, 42
VAL_SPLIT = 0.1

# 子图相关配置
USE_SUBGRAPHS = True    # 为False时，使用原始图数据，NUM_HOPS不起作用
NUM_HOPS = 4            # k-hop邻居数（3, 4, 5, 6）, 推荐（3, 4）

