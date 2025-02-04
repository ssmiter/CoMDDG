"""dataset_process/enhanced_feature_gen.py"""
import numpy as np
import warnings
import logging

from Bio.Align import substitution_matrices
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.vectors import calc_dihedral
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from config import FeatureConfig
from functools import lru_cache



# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 优化one-hot编码计算
AA_DICT = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}

# 氨基酸理化性质数据
# Kyte-Doolittle疏水性指数
kd_hydrophobicity_scale_raw = {
    'ALA': 1.8, 'CYS': 2.5, 'ASP': -3.5, 'GLU': -3.5, 'PHE': 2.8,
    'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LYS': -3.9, 'LEU': 3.8,
    'MET': 1.9, 'ASN': -3.5, 'PRO': -1.6, 'GLN': -3.5, 'ARG': -4.5,
    'SER': -0.8, 'THR': -0.7, 'VAL': 4.2, 'TRP': -0.9, 'TYR': -1.3
}

# 氨基酸体积
residue_volumes_raw = {
    'ALA': 88.6, 'CYS': 108.5, 'ASP': 111.1, 'GLU': 138.4, 'PHE': 189.9,
    'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7, 'LYS': 168.6, 'LEU': 166.7,
    'MET': 162.9, 'ASN': 114.1, 'PRO': 112.7, 'GLN': 143.8, 'ARG': 173.4,
    'SER': 89.0, 'THR': 116.1, 'VAL': 140.0, 'TRP': 227.8, 'TYR': 193.6
}

# 氨基酸极性
residue_polarity_raw = {
    'ALA': 0.0, 'CYS': 1.48, 'ASP': 40.7, 'GLU': 40.7, 'PHE': 0.35,
    'GLY': 0.0, 'HIS': 3.53, 'ILE': 0.13, 'LYS': 49.5, 'LEU': 0.13,
    'MET': 1.43, 'ASN': 3.38, 'PRO': 1.58, 'GLN': 3.53, 'ARG': 52.0,
    'SER': 1.67, 'THR': 1.66, 'VAL': 0.13, 'TRP': 2.1, 'TYR': 1.61
}

# pKa值
residue_pka_raw = {
    'ALA': None, 'CYS': 8.37, 'ASP': 3.91, 'GLU': 4.15, 'PHE': None,
    'GLY': None, 'HIS': 6.04, 'ILE': None, 'LYS': 10.67, 'LEU': None,
    'MET': None, 'ASN': None, 'PRO': None, 'GLN': None, 'ARG': 12.10,
    'SER': None, 'THR': None, 'VAL': None, 'TRP': None, 'TYR': 10.46
}

# 氨基酸电荷 (pH 7.4)
residue_charge = {
    'ALA': 0.0, 'CYS': 0.0, 'ASP': -1.0, 'GLU': -1.0, 'PHE': 0.0,
    'GLY': 0.0, 'HIS': 0.1, 'ILE': 0.0, 'LYS': 1.0, 'LEU': 0.0,
    'MET': 0.0, 'ASN': 0.0, 'PRO': 0.0, 'GLN': 0.0, 'ARG': 1.0,
    'SER': 0.0, 'THR': 0.0, 'VAL': 0.0, 'TRP': 0.0, 'TYR': 0.0
}

# 最大溶剂可及表面积 (Max ASA)
max_acc = {
    'ALA': 121.0, 'ARG': 265.0, 'ASN': 187.0, 'ASP': 187.0, 'CYS': 148.0,
    'GLN': 214.0, 'GLU': 214.0, 'GLY': 97.0, 'HIS': 216.0, 'ILE': 195.0,
    'LEU': 191.0, 'LYS': 230.0, 'MET': 203.0, 'PHE': 228.0, 'PRO': 154.0,
    'SER': 143.0, 'THR': 163.0, 'TRP': 264.0, 'TYR': 255.0, 'VAL': 165.0
}

# 柔性指数 (Flexibility index)
# 来源：Vihinen et al. (1994) Proteins, 19, 141-149
flexibility_indices = {
    'ALA': 0.360, 'CYS': 0.350, 'ASP': 0.510, 'GLU': 0.500, 'PHE': 0.310,
    'GLY': 0.540, 'HIS': 0.320, 'ILE': 0.460, 'LYS': 0.470, 'LEU': 0.370,
    'MET': 0.300, 'ASN': 0.460, 'PRO': 0.510, 'GLN': 0.490, 'ARG': 0.530,
    'SER': 0.510, 'THR': 0.440, 'VAL': 0.390, 'TRP': 0.310, 'TYR': 0.420
}

# Beta折叠倾向性
# 来源：Chou & Fasman (1974) Biochemistry, 13, 222-245
beta_propensities = {
    'ALA': 0.83, 'CYS': 1.19, 'ASP': 0.54, 'GLU': 0.37, 'PHE': 1.38,
    'GLY': 0.75, 'HIS': 0.87, 'ILE': 1.60, 'LYS': 0.74, 'LEU': 1.30,
    'MET': 1.05, 'ASN': 0.89, 'PRO': 0.55, 'GLN': 1.10, 'ARG': 0.93,
    'SER': 0.75, 'THR': 1.19, 'VAL': 1.70, 'TRP': 1.37, 'TYR': 1.47
}

# Turn倾向性
# 来源：Chou & Fasman (1974) Biochemistry, 13, 222-245
turn_propensities = {
    'ALA': 0.66, 'CYS': 1.19, 'ASP': 1.46, 'GLU': 0.74, 'PHE': 0.60,
    'GLY': 1.56, 'HIS': 0.95, 'ILE': 0.47, 'LYS': 1.01, 'LEU': 0.59,
    'MET': 0.60, 'ASN': 1.56, 'PRO': 1.52, 'GLN': 0.98, 'ARG': 0.95,
    'SER': 1.43, 'THR': 0.96, 'VAL': 0.50, 'TRP': 0.96, 'TYR': 1.14
}

# 暴露倾向性
# 来源：Janin (1979) Nature, 277, 491-492
exposure_propensities = {
    'ALA': 0.48, 'CYS': 0.32, 'ASP': 0.81, 'GLU': 0.93, 'PHE': 0.23,
    'GLY': 0.51, 'HIS': 0.66, 'ILE': 0.34, 'LYS': 0.97, 'LEU': 0.40,
    'MET': 0.38, 'ASN': 0.82, 'PRO': 0.78, 'GLN': 0.81, 'ARG': 0.84,
    'SER': 0.70, 'THR': 0.71, 'VAL': 0.40, 'TRP': 0.31, 'TYR': 0.42
}

# Alpha螺旋倾向性
# 来源：Chou & Fasman (1974) Biochemistry, 13, 222-245
alpha_helix_propensities = {
    'ALA': 1.42, 'CYS': 0.70, 'ASP': 1.01, 'GLU': 1.51, 'PHE': 1.13,
    'GLY': 0.57, 'HIS': 1.00, 'ILE': 1.08, 'LYS': 1.16, 'LEU': 1.21,
    'MET': 1.45, 'ASN': 0.67, 'PRO': 0.57, 'GLN': 1.11, 'ARG': 0.98,
    'SER': 0.77, 'THR': 0.83, 'VAL': 1.06, 'TRP': 1.08, 'TYR': 0.69
}

# 归一化后的字典 (使用原始名称)
kd_hydrophobicity_scale = {
    'ALA': 0.700, 'CYS': 0.778, 'ASP': 0.111, 'GLU': 0.111, 'PHE': 0.811,
    'GLY': 0.456, 'HIS': 0.144, 'ILE': 1.000, 'LYS': 0.067, 'LEU': 0.922,
    'MET': 0.711, 'ASN': 0.111, 'PRO': 0.322, 'GLN': 0.111, 'ARG': 0.000,
    'SER': 0.411, 'THR': 0.422, 'VAL': 0.967, 'TRP': 0.400, 'TYR': 0.356
}

residue_volumes = {
    'ALA': 0.168, 'CYS': 0.285, 'ASP': 0.301, 'GLU': 0.461, 'PHE': 0.764,
    'GLY': 0.001, 'HIS': 0.548, 'ILE': 0.628, 'LYS': 0.639, 'LEU': 0.628,
    'MET': 0.605, 'ASN': 0.318, 'PRO': 0.310, 'GLN': 0.493, 'ARG': 0.667,
    'SER': 0.171, 'THR': 0.330, 'VAL': 0.471, 'TRP': 0.987, 'TYR': 0.786
}

residue_polarity = {
    'ALA': 0.000, 'CYS': 0.028, 'ASP': 0.783, 'GLU': 0.783, 'PHE': 0.007,
    'GLY': 0.000, 'HIS': 0.068, 'ILE': 0.003, 'LYS': 0.952, 'LEU': 0.003,
    'MET': 0.028, 'ASN': 0.065, 'PRO': 0.030, 'GLN': 0.068, 'ARG': 1.000,
    'SER': 0.032, 'THR': 0.032, 'VAL': 0.003, 'TRP': 0.040, 'TYR': 0.031
}

residue_pka = {
    'ALA': 0.000, 'CYS': 0.541, 'ASP': 0.046, 'GLU': 0.072, 'PHE': 0.000,
    'GLY': 0.000, 'HIS': 0.282, 'ILE': 0.000, 'LYS': 0.797, 'LEU': 0.000,
    'MET': 0.000, 'ASN': 0.000, 'PRO': 0.000, 'GLN': 0.000, 'ARG': 0.956,
    'SER': 0.000, 'THR': 0.000, 'VAL': 0.000, 'TRP': 0.000, 'TYR': 0.773
}

# residue_charge = {
#     'ALA': 0.500, 'CYS': 0.500, 'ASP': 0.000, 'GLU': 0.000, 'PHE': 0.500,
#     'GLY': 0.500, 'HIS': 0.550, 'ILE': 0.500, 'LYS': 1.000, 'LEU': 0.500,
#     'MET': 0.500, 'ASN': 0.500, 'PRO': 0.500, 'GLN': 0.500, 'ARG': 1.000,
#     'SER': 0.500, 'THR': 0.500, 'VAL': 0.500, 'TRP': 0.500, 'TYR': 0.500
# }

# max_acc = {  # 归一化到[0,1]，使用最大值265作为归一化因子
#     'ALA': 0.457, 'ARG': 1.000, 'ASN': 0.706, 'ASP': 0.706, 'CYS': 0.558,
#     'GLN': 0.808, 'GLU': 0.808, 'GLY': 0.366, 'HIS': 0.815, 'ILE': 0.736,
#     'LEU': 0.721, 'LYS': 0.868, 'MET': 0.766, 'PHE': 0.860, 'PRO': 0.581,
#     'SER': 0.540, 'THR': 0.615, 'TRP': 0.996, 'TYR': 0.962, 'VAL': 0.623
# }

# flexibility_indices = {
#     'ALA': 0.250, 'CYS': 0.208, 'ASP': 0.875, 'GLU': 0.833, 'PHE': 0.042,
#     'GLY': 1.000, 'HIS': 0.083, 'ILE': 0.667, 'LYS': 0.708, 'LEU': 0.292,
#     'MET': 0.000, 'ASN': 0.667, 'PRO': 0.875, 'GLN': 0.792, 'ARG': 0.958,
#     'SER': 0.875, 'THR': 0.583, 'VAL': 0.375, 'TRP': 0.042, 'TYR': 0.500
# }

# Beta折叠倾向性 (与beta_propensities相同但保留以保持一致性)
beta_sheet_propensities = beta_propensities

# Coil倾向性
# 计算为1减去alpha和beta倾向性的平均值
coil_propensities = {
    aa: 1.0 - (alpha_helix_propensities[aa] + beta_propensities[aa]) / 2
    for aa in alpha_helix_propensities.keys()
}

# 导入BLOSUM62矩阵
blosum62 = substitution_matrices.load("BLOSUM62")

# 忽略BioPython的警告
warnings.filterwarnings('ignore', category=PDBConstructionWarning)

@lru_cache(maxsize=20)
def get_onehot(residue_name):
    return np.array(AA_DICT.get(residue_name, np.zeros(20)))


def calculate_node_features(residue, sequence, structure, pssm, conservation_score):
    """改进的节点特征计算,主入口函数"""
    try:
        # 初始化特征数组
        features = np.zeros(FeatureConfig.get_total_node_dim())
        slices = FeatureConfig.get_node_feature_slices()

        # 分别计算各个特征并打印维度
        # 1. one-hot编码 (20维)
        one_hot = get_onehot(residue.get_resname())

        # 2. 序列属性特征 (12维)
        seq_props = calculate_sequence_properties(residue)

        # 3. 结构特征 (14维)
        # 暂时不使用结构特征
        # struct_feats = calculate_structure_features_safe(residue, structure)

        # 4. 进化特征 (21维)
        evol_feats = calculate_evolution_features(residue, pssm, conservation_score, len(sequence))

        # 将特征放入对应的切片位置
        features[slices['one_hot']] = one_hot
        features[slices['seq_properties']] = seq_props
        # features[slices['structure']] = struct_feats
        features[slices['evolution']] = evol_feats

        return features

    except Exception as e:
        logging.error(f"Error calculating features for residue {residue}: {str(e)}")
        return np.zeros(FeatureConfig.get_total_node_dim())


def normalize_features(features):
    """改进的特征归一化函数"""
    slices = FeatureConfig.get_node_feature_slices()
    normalized = features.copy()

    # 1. One-hot编码 (不需要归一化)
    # one_hot切片 = slices['one_hot']

    # 2. 序列属性归一化
    seq_props_slice = slices['seq_properties']
    # 物理化学性质归一化
    seq_props = normalized[seq_props_slice]
    # 疏水性 (-4.5到4.5)
    seq_props[0] = (seq_props[0] + 4.5) / 9.0
    # 体积 (0-230)
    seq_props[1] = seq_props[1] / 230.0
    # 极性 (0-52)
    seq_props[2] = seq_props[2] / 52.0
    # pKa (0-14)
    seq_props[3] = seq_props[3] / 14.0 if seq_props[3] != 0 else 0
    # 电荷 (-1到1)
    seq_props[4] = (seq_props[4] + 1) / 2.0
    # 其他序列属性已经在0-1范围内
    normalized[seq_props_slice] = seq_props

    # 3. 结构特征归一化
    # 暂时不使用结构特征
    # struct_slice = slices['structure']
    # struct_feats = normalized[struct_slice]
    # # ASA归一化
    # struct_feats[4] = struct_feats[4] / 1000.0 if struct_feats[4] != 0 else 0
    # # B-factor归一化
    # struct_feats[5] = (struct_feats[5] - 30) / 30 if struct_feats[5] != 0 else 0
    # normalized[struct_slice] = struct_feats

    # 4. 进化特征 (已经在0-1范围内)

    # 最后的检查和裁剪
    normalized = np.clip(normalized, 0, 1)

    return normalized


def batch_normalize_features(features_batch):
    """批量归一化特征"""
    normalized_batch = np.zeros_like(features_batch)
    for i in range(len(features_batch)):
        try:
            normalized_batch[i] = normalize_features(features_batch[i])
        except Exception as e:
            print(f"Error normalizing features for batch {i}: {str(e)}")
            normalized_batch[i] = features_batch[i]  # 如果归一化失败，保持原值
    return normalized_batch


# 添加用于验证归一化效果的函数
def validate_normalization(features_batch, normalized_batch):
    """验证归一化的效果"""
    print("\nNormalization validation:")
    slices = FeatureConfig.get_node_feature_slices()

    for name, slice_obj in slices.items():
        original = features_batch[:, slice_obj]
        normalized = normalized_batch[:, slice_obj]

        print(f"\n{name}:")
        print(f"Original - Min: {np.min(original):.4f}, Max: {np.max(original):.4f}")
        print(f"Normalized - Min: {np.min(normalized):.4f}, Max: {np.max(normalized):.4f}")


def calculate_structure_features_safe(residue, structure, radius=10.0):
    """改进的结构特征计算，确保14维输出"""
    try:
        # 初始化14维特征向量
        structure_features = np.zeros(FeatureConfig.STRUCTURE_DIM)
        current_index = 0

        # 1. 二级结构 (3维)
        ss = calculate_secondary_structure_safe(residue, structure)
        structure_features[current_index:current_index + 3] = ss
        current_index += 3

        # 2. 溶剂可及性 (2维)
        rsa = calculate_relative_solvent_accessibility_safe(residue, structure)
        asa = calculate_absolute_solvent_accessibility_safe(residue, structure)
        structure_features[current_index:current_index + 2] = [
            min(max(rsa, 0), 1),
            min(asa, 1000) / 1000  # 归一化ASA
        ]
        current_index += 2

        # 3. B-factor (1维)
        if residue.has_id('CA'):
            b_factor = residue['CA'].get_bfactor()
            structure_features[current_index] = max(min(b_factor / 100.0, 1.0), 0.0)
        current_index += 1

        # 4. 局部环境特征 (8维)
        # - 接触密度 (1维)
        # - 氢键信息 (2维)
        # - 到突变位点的距离 (1维)
        # - 局部结构密度 (1维)
        # - 相对接触序 (1维)
        # - 原子深度 (2维)
        if residue.has_id('CA'):
            local_features = calculate_local_environment_safe(residue, structure, radius)
            structure_features[current_index:current_index + 8] = local_features

        assert len(structure_features) == FeatureConfig.STRUCTURE_DIM, \
            f"Structure features dimension mismatch: {len(structure_features)} != {FeatureConfig.STRUCTURE_DIM}"

        return structure_features

    except Exception as e:
        logging.error(f"Error in calculate_structure_features_safe: {str(e)}")
        return np.zeros(FeatureConfig.STRUCTURE_DIM)


# 优化结构特征计算，使用缓存
@lru_cache(maxsize=1024)
def calculate_structure_features_cached(residue_id, structure_id, radius=10.0):
    """使用缓存的结构特征计算"""
    residue = structure_id[0][residue_id[0]][residue_id[1]]
    if not residue.has_id('CA'):
        return np.zeros(16)

    structure_features = np.zeros(16)

    # 通过预计算CA坐标加速距离计算
    ca_coords = {}
    for chain in structure_id[0]:
        for res in chain:
            if 'CA' in res:
                ca_coords[(chain.id, res.id)] = res['CA'].get_coord()

    # 计算接触密度
    center = ca_coords[(residue_id[0], residue_id[1])]
    contacts = 0
    total = 0
    for coord in ca_coords.values():
        dist = np.linalg.norm(center - coord)
        if dist < radius:
            contacts += 1
        total += 1
    structure_features[0] = contacts / max(total, 1)

    # 其他结构特征的计算...

    return structure_features


def calculate_secondary_structure_safe(residue, structure):
    """安全版本的二级结构计算"""
    try:
        # 默认二级结构编码 (coil)
        ss_encoding = np.array([0, 0, 1])

        # 获取structure的ID和文件路径
        structure_id = structure.get_id()
        if hasattr(structure, 'get_filename'):
            pdb_file = structure.get_filename()
        else:
            return ss_encoding

        # 使用dssp_dict_from_pdb_file而不是DSSP类
        dssp_dict = dssp_dict_from_pdb_file(pdb_file)[0]

        # 获取残基的key
        chain_id = residue.get_parent().id
        res_id = residue.get_id()
        key = (chain_id, res_id)

        if key in dssp_dict:
            ss_type = dssp_dict[key][2]
            if ss_type in ['H', 'G', 'I']:  # Helix
                ss_encoding = np.array([1, 0, 0])
            elif ss_type in ['E', 'B']:  # Sheet
                ss_encoding = np.array([0, 1, 0])
            # else保持为coil

        return ss_encoding

    except Exception as e:
        print(f"Error in calculate_secondary_structure_safe: {str(e)}")
        return np.array([0, 0, 1])  # 默认为coil


def calculate_structure_density_safe(residue, structure, radius=10.0):
    """
    计算残基局部结构密度

    Args:
        residue: Biopython残基对象
        structure: 结构对象
        radius: 搜索半径（埃米）

    Returns:
        float: 局部结构密度（原子数/体积）
    """
    try:
        if not residue.has_id('CA'):
            return 0.0

        center = residue['CA'].get_coord()
        atom_count = 0

        # 计算在搜索半径内的原子数
        for atom in structure.get_atoms():
            dist = np.linalg.norm(atom.get_coord() - center)
            if dist <= radius:
                atom_count += 1

        # 计算球形搜索空间的体积
        volume = (4 / 3) * np.pi * radius ** 3

        # 计算密度（原子数/体积）
        density = atom_count / volume if volume > 0 else 0.0

        # 归一化密度（通过除以典型的最大密度值）
        typical_max_density = 0.1  # 典型的最大原子密度值（原子/Å³）
        normalized_density = min(density / typical_max_density, 1.0)

        return normalized_density

    except Exception as e:
        logging.error(f"Error calculating structure density for residue {residue}: {e}")
        return 0.0


def calculate_relative_solvent_accessibility_safe(residue, structure):
    """安全版本的RSA计算"""
    try:
        sr = ShrakeRupley()
        sr.compute(structure, level="R")
        residue_max_acc = max_acc.get(residue.get_resname(), 200.0)
        return residue.sasa / residue_max_acc
    except Exception as e:
        return 0.0


def calculate_absolute_solvent_accessibility_safe(residue, structure):
    """安全版本的ASA计算"""
    try:
        sr = ShrakeRupley()
        sr.compute(structure, level="R")
        return residue.sasa
    except Exception as e:
        return 0.0


def is_valid_vector(v, epsilon=1e-6):
    """检查向量是否有效"""
    try:
        norm = v.norm()
        return norm > epsilon
    except Exception:
        return False


def safe_calc_dihedral(v1, v2, v3, v4, epsilon=1e-10):
    """更稳定的二面角计算"""
    try:
        # 检查向量的有效性
        if not all(is_valid_vector(v, epsilon) for v in [v1, v2, v3, v4]):
            return 0.0

        # 使用Bio.PDB的内置方法计算二面角
        try:
            angle = calc_dihedral(v1, v2, v3, v4)
            # 检查结果是否有效
            if np.isnan(angle) or np.isinf(angle):
                return 0.0
            return float(angle)
        except Exception as e:
            logging.debug(f"Error in dihedral calculation: {e}")
            return 0.0

    except Exception as e:
        logging.debug(f"Error in dihedral calculation: {e}")
        return 0.0


def calculate_backbone_angles_safe(residue):
    """更安全的主链二面角计算"""
    try:
        phi = psi = 0.0

        if (residue.has_id('N') and residue.has_id('CA') and
                residue.has_id('C') and 'O' in residue):

            try:
                prev_res = list(residue.get_parent().get_residues())[-1]
                next_res = list(residue.get_parent().get_residues())[1]

                # 计算phi角
                if prev_res.has_id('C'):
                    vectors_phi = [
                        prev_res['C'].get_vector(),
                        residue['N'].get_vector(),
                        residue['CA'].get_vector(),
                        residue['C'].get_vector()
                    ]
                    phi = safe_calc_dihedral(*vectors_phi)

                # 计算psi角
                if next_res.has_id('N'):
                    vectors_psi = [
                        residue['N'].get_vector(),
                        residue['CA'].get_vector(),
                        residue['C'].get_vector(),
                        next_res['N'].get_vector()
                    ]
                    psi = safe_calc_dihedral(*vectors_psi)

            except Exception as e:
                logging.debug(f"Error in backbone angle calculation for residue {residue}: {e}")
                return 0.0, 0.0

        return float(phi), float(psi)

    except Exception as e:
        logging.error(f"Error in calculate_backbone_angles_safe: {str(e)}")
        return 0.0, 0.0


def calculate_local_environment_safe(residue, structure, radius=10.0):
    """改进的局部环境特征计算（8维）"""
    try:
        local_features = np.zeros(8)

        # 1. 局部接触密度 - 正确传递参数
        residue_id = get_residue_identifier(residue)
        local_features[0] = calculate_contact_density_safe(residue_id, structure, radius)

        # 2. 氢键信息（2维：主链氢键和侧链氢键）
        bb_hbonds, sc_hbonds = calculate_hbonds_safe(residue, structure)
        local_features[1:3] = [bb_hbonds, sc_hbonds]

        # 3. 到突变位点的距离
        local_features[3] = calculate_mutation_distance_safe(residue, structure)

        # 4. 局部结构密度
        local_features[4] = calculate_structure_density_safe(residue, structure, radius)

        # 5. 相对接触序
        local_features[5] = calculate_relative_contact_order_safe(residue, structure)

        # 6. 原子深度（2维）
        mc_depth, sc_depth = calculate_atom_depth_safe(residue, structure)
        local_features[6:8] = [mc_depth, sc_depth]

        return local_features
    except Exception as e:
        logging.error(f"Error in calculate_local_environment_safe: {str(e)}")
        return np.zeros(8)


# def calculate_contact_density_safe(residue, structure, radius):
#     """安全版本的接触密度计算"""
#     try:
#         if not residue.has_id('CA'):
#             return 0.0
#
#         center = residue['CA'].get_coord()
#         ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == 'CA']
#         contacts = sum(1 for atom in ca_atoms
#                        if atom != residue['CA'] and
#                        np.linalg.norm(center - atom.get_coord()) < radius)
#
#         return contacts / max(len(ca_atoms), 1)
#     except Exception as e:
#         return 0.0


def calculate_hbonds_safe(residue, structure, distance_cutoff=3.5, angle_cutoff=120):
    """
    计算残基的氢键特征

    Args:
        residue: 目标残基
        structure: 结构对象
        distance_cutoff: 氢键距离阈值(Å)
        angle_cutoff: 氢键角度阈值(度)

    Returns:
        tuple: (backbone_hbonds, sidechain_hbonds) 主链和侧链的氢键数
    """
    try:
        backbone_hbonds = 0
        sidechain_hbonds = 0

        # 获取目标残基的主链原子
        if not (residue.has_id('N') and residue.has_id('O')):
            return 0.0, 0.0

        target_n = residue['N'].get_vector()
        target_o = residue['O'].get_vector()

        # 遍历所有其他残基寻找潜在的氢键
        for other_res in structure.get_residues():
            if other_res != residue and other_res.has_id('N') and other_res.has_id('O'):
                # 主链-主链氢键检查
                other_n = other_res['N'].get_vector()
                other_o = other_res['O'].get_vector()

                # N-H...O氢键
                no_dist = (target_n - other_o).norm()
                if no_dist < distance_cutoff:
                    backbone_hbonds += 1

                # O...H-N氢键
                on_dist = (target_o - other_n).norm()
                if on_dist < distance_cutoff:
                    backbone_hbonds += 1

                # 侧链-主链氢键检查
                sidechain_atoms = [atom for atom in residue if atom.name not in ('N', 'CA', 'C', 'O')]
                for atom in sidechain_atoms:
                    if atom.element != 'H':  # 跳过氢原子
                        atom_vec = atom.get_vector()
                        # 到其他残基主链的距离
                        dist_n = (atom_vec - other_n).norm()
                        dist_o = (atom_vec - other_o).norm()
                        if dist_n < distance_cutoff or dist_o < distance_cutoff:
                            sidechain_hbonds += 1

        # 标准化氢键数
        max_backbone_hbonds = 4.0  # 典型的主链氢键最大数
        max_sidechain_hbonds = 8.0  # 典型的侧链氢键最大数

        return (min(backbone_hbonds / max_backbone_hbonds, 1.0),
                min(sidechain_hbonds / max_sidechain_hbonds, 1.0))

    except Exception as e:
        print(f"Error in calculate_hbonds_safe: {str(e)}")
        return 0.0, 0.0


def calculate_mutation_distance_safe(residue, structure):
    """安全版本的突变位点距离计算"""
    return 0.0


def get_residue_identifier(residue):
    """获取残基的唯一标识符（可哈希）"""
    return (residue.get_parent().id, residue.id[1])  # 返回(链ID, 残基编号)


@lru_cache(maxsize=1024)
def calculate_contact_density_safe(residue_id, structure_id, radius=10.0):
    """使用缓存优化的接触密度计算"""
    try:
        chain_id, res_num = residue_id
        residue = structure_id[0][chain_id][res_num]

        if not residue.has_id('CA'):
            return 0.0

        center = residue['CA'].get_coord()
        count = 0
        total = 0

        for other_res in structure_id[0].get_residues():
            if other_res.has_id('CA') and other_res != residue:
                dist = np.linalg.norm(center - other_res['CA'].get_coord())
                if dist < radius:
                    count += 1
                total += 1

        return count / max(total, 1)
    except Exception as e:
        logging.error(f"Error in contact density calculation for residue {residue_id}: {e}")
        return 0.0


def calculate_relative_contact_order_safe(residue, structure):
    """计算残基的相对接触序"""
    try:
        contacts = 0
        total_sequence_separation = 0

        if not residue.has_id('CA'):
            return 0.0

        residue_pos = residue.get_id()[1]
        ca_coord = residue['CA'].get_coord()

        for other_res in structure.get_residues():
            if other_res != residue and other_res.has_id('CA'):
                other_ca = other_res['CA'].get_coord()
                distance = np.linalg.norm(ca_coord - other_ca)

                if distance < 8.0:  # 典型的接触距离阈值
                    contacts += 1
                    sequence_separation = abs(residue_pos - other_res.get_id()[1])
                    total_sequence_separation += sequence_separation

        if contacts > 0:
            return total_sequence_separation / (contacts * len(list(structure.get_residues())))
        return 0.0

    except Exception as e:
        print(f"Error in calculate_relative_contact_order_safe: {str(e)}")
        return 0.0


def calculate_atom_depth_safe(residue, structure):
    """
    计算残基的原子深度
    返回主链和侧链原子的平均深度
    """
    try:
        if not residue.has_id('CA'):
            return 0.0, 0.0

        # 获取所有原子的坐标
        all_coords = np.array([atom.get_coord() for atom in structure.get_atoms()])

        # 分离主链和侧链原子
        backbone_atoms = [atom for atom in residue if atom.name in ('N', 'CA', 'C', 'O')]
        sidechain_atoms = [atom for atom in residue if atom.name not in ('N', 'CA', 'C', 'O')]

        def calculate_depth(atoms):
            if not atoms:
                return 0.0
            depths = []
            for atom in atoms:
                coord = atom.get_coord()
                # 计算到所有其他原子的最小距离
                distances = np.linalg.norm(all_coords - coord, axis=1)
                # 取最小的非零距离作为深度
                min_dist = np.min(distances[distances > 0]) if len(distances[distances > 0]) > 0 else 0
                depths.append(min_dist)
            return np.mean(depths)

        mc_depth = calculate_depth(backbone_atoms)
        sc_depth = calculate_depth(sidechain_atoms) if sidechain_atoms else mc_depth

        # 标准化深度值（假设最大深度为10Å）
        max_depth = 10.0
        return mc_depth / max_depth, sc_depth / max_depth

    except Exception as e:
        print(f"Error in calculate_atom_depth_safe: {str(e)}")
        return 0.0, 0.0


def calculate_sequence_properties(residue):
    """计算序列相关特征"""
    res_name = residue.get_resname()

    # 疏水性指数 (Kyte-Doolittle)
    hydrophobicity = kd_hydrophobicity_scale.get(res_name, 0.0)

    # 体积
    volume = residue_volumes.get(res_name, 0.0)

    # 极性
    polarity = residue_polarity.get(res_name, 0.0)

    # pKa
    pka = residue_pka.get(res_name, 0.0) if residue_pka.get(res_name) is not None else 0.0

    # 电荷
    charge = residue_charge.get(res_name, 0.0)

    # 其他各种倾向性
    flexibility_index = flexibility_indices.get(res_name, 0.0)
    beta_propensity = beta_propensities.get(res_name, 0.0)
    turn_propensity = turn_propensities.get(res_name, 0.0)
    exposure_propensity = exposure_propensities.get(res_name, 0.0)
    alpha_helix_propensity = alpha_helix_propensities.get(res_name, 0.0)
    beta_sheet_propensity = beta_sheet_propensities.get(res_name, 0.0)
    coil_propensity = coil_propensities.get(res_name, 0.0)

    return np.array([
        hydrophobicity, volume, polarity, pka, charge,
        flexibility_index, beta_propensity, turn_propensity,
        exposure_propensity, alpha_helix_propensity,
        beta_sheet_propensity, coil_propensity
    ])


def analyze_pssm_distribution(pssm_data):
    """分析PSSM数据的分布情况"""
    if isinstance(pssm_data, np.ndarray):
        print("\nPSSM Distribution Analysis:")
        print(f"Shape: {pssm_data.shape}")
        print(f"Raw Mean: {np.mean(pssm_data):.4f}")
        print(f"Raw Std: {np.std(pssm_data):.4f}")
        print(f"Raw Range: [{np.min(pssm_data):.4f}, {np.max(pssm_data):.4f}]")
        print("\nPercentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"{p}th percentile: {np.percentile(pssm_data, p):.4f}")


def normalize_pssm(pssm_scores, res_idx, method='robust_sigmoid', scale_factor=1.2, stretch_factor=2.0):
    """
    改进的PSSM归一化, 增加区分度

    Args:
        pssm_scores: 原始PSSM得分
        method: 归一化方法，可选：
               'robust_sigmoid' - 使用robust scaling后的sigmoid
               'minmax' - 简单的min-max归一化
               'zscore' - Z-score归一化
               'scaled_sigmoid' - 缩放后的sigmoid归一化
        scale_factor：主要影响分布的集中度,降低scale_factor会使得数据分布更分散
        stretch_factor：主要影响标准差,增加stretch_factor会直接提高标准差

    Returns:
        归一化后的PSSM得分
    """
    try:
        if not isinstance(pssm_scores, np.ndarray):
            return np.zeros(20)  # 默认返回20维零向量

        if method == 'simple':
            return np.clip(pssm_scores[res_idx], -10, 10) / 10.0

        elif method == 'robust_sigmoid':
            # 使用四分位数范围(IQR)进行缩放
            # 1. Robust scaling
            q1 = np.percentile(pssm_scores, 25)
            q3 = np.percentile(pssm_scores, 75)
            iqr = q3 - q1 if q3 > q1 else 1.0

            # 2. 缩放
            scaled_scores = (pssm_scores - np.median(pssm_scores)) / (iqr / scale_factor)

            # 3. 拉伸
            scaled_scores = scaled_scores * stretch_factor
            return 1 / (1 + np.exp(-scaled_scores))

        elif method == 'minmax':
            # 限制极值后进行min-max归一化
            clipped_scores = np.clip(pssm_scores, -10, 10)
            min_val = np.min(clipped_scores)
            max_val = np.max(clipped_scores)
            if max_val == min_val:
                return np.zeros_like(pssm_scores)
            return (clipped_scores - min_val) / (max_val - min_val)

        elif method == 'zscore':
            # 标准差归一化，处理异常值
            mean = np.mean(pssm_scores)
            std = np.std(pssm_scores)
            if std == 0:
                return np.zeros_like(pssm_scores)
            z_scores = (pssm_scores - mean) / std
            return 1 / (1 + np.exp(-z_scores))  # 用sigmoid函数压缩到(0,1)

        elif method == 'scaled_sigmoid':
            # 缩放因子的sigmoid归一化
            scale_factor = 5.0  # 可调整的缩放因子
            return 1 / (1 + np.exp(-pssm_scores / scale_factor))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    except Exception as e:
        logging.error(f"Error in PSSM normalization: {e}")
        return np.zeros_like(pssm_scores)


def calculate_evolution_features(residue, pssm, conservation_score, seq_length):
    """改进的进化特征计算函数"""
    try:
        # res_idx = residue.get_id()[1] - 1  # 转换为0-based索引
        res_idx = residue.get_id()[1]
        if res_idx < 0 or res_idx >= seq_length:
            logging.debug(
                f"Residue index {res_idx} out of sequence bounds ({seq_length})"
            )
            return np.zeros(21)

        # 处理PSSM特征
        if isinstance(pssm, np.ndarray) and pssm.shape[0] > res_idx:
            pssm_scores = pssm[res_idx]
            # 归一化PSSM得分
            pssm_scores = np.clip(pssm_scores, -10, 10)
            pssm_scores = (pssm_scores + 10) / 20
        else:
            logging.debug(f"Invalid PSSM data for residue {residue.get_id()}")
            pssm_scores = np.zeros(20)

        # 处理Conservation得分
        if isinstance(conservation_score, np.ndarray) and conservation_score.shape[0] > res_idx:
            cons_score = conservation_score[res_idx]
        else:
            cons_score = 0.0

        return np.concatenate([pssm_scores, [cons_score]])

    except Exception as e:
        logging.error(f"Error calculating evolution features for residue {residue.get_id()}: {e}")
        return np.zeros(21)


def create_sequence_mapping(sequence, structure):
    """创建序列到结构的映射"""
    mapping = {}
    seq_idx = 0

    for chain in structure[0]:
        for residue in chain:
            if residue.get_id()[0] == ' ':  # 只处理标准残基
                mapping[(chain.id, residue.get_id())] = seq_idx
                seq_idx += 1

    return mapping
