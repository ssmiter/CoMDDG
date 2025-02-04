"""enhanced_graph_gen_structure_simplified.py - Simplified graph generation with batch checkpoints"""
import os
import sys
import logging
import pickle
import numpy as np
from pathlib import Path
import glob

from Bio.PDB import PDBParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from dataset_process.config import FeatureConfig
# Import other necessary functions from your existing code
from enhanced_feature_gen import calculate_structure_features_safe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('graph_generation.log')
    ]
)

def find_latest_batch(output_path: str) -> Tuple[int, int]:
    """
    查找最新的批处理文件，返回已处理的样本数和下一个批次号
    """
    pattern = output_path.replace('.pkl', '_temp_*.pkl')
    batch_files = glob.glob(pattern)
    
    if not batch_files:
        return 0, 0
        
    # 从文件名提取批次号
    batch_numbers = []
    for file in batch_files:
        try:
            num = int(file.split('_temp_')[-1].replace('.pkl', ''))
            batch_numbers.append(num)
        except ValueError:
            continue
            
    if not batch_numbers:
        return 0, 0
        
    latest_batch = max(batch_numbers)
    return latest_batch, latest_batch // 50


def validate_features(features: np.ndarray, expected_dim: int, name: str) -> bool:
    """验证特征维度并记录日志"""
    if features is None:
        logging.error(f"{name} features is None")
        return False

    if features.shape[1] != expected_dim:
        logging.error(f"Invalid {name} feature dimension: {features.shape[1]}, expected {expected_dim}")
        return False

    if np.isnan(features).any() or np.isinf(features).any():
        logging.error(f"{name} features contain NaN or Inf values")
        return False

    return True

def find_mutation_position(wild_features, mutant_features):
    """
    通过比较one-hot编码找到突变位点

    Args:
        wild_features: 野生型特征数组
        mutant_features: 突变型特征数组

    Returns:
        mutation_pos: 突变位点的索引
        mutation_info: 包含野生型和突变型氨基酸信息的字典
    """
    for i in range(len(wild_features)):
        # 只比较前20维的one-hot编码部分(氨基酸类型)
        if not np.array_equal(wild_features[i][:20], mutant_features[i][:20]):
            wild_aa = np.argmax(wild_features[i][:20])
            mutant_aa = np.argmax(mutant_features[i][:20])
            return i, {
                'wild_aa': wild_aa,
                'mutant_aa': mutant_aa,
                'position': i
            }
    return None, None

def process_protein(args) -> Tuple[Optional[Dict], Optional[Dict]]:
    """处理单个蛋白质对，确保残基对齐并验证特征维度"""
    wild_data, mutant_data, ddg, wild_name, mutant_name, pdb_dir = args

    logging.info(f"Processing protein pair: {wild_name} -> {mutant_name}")

    try:
        parser = PDBParser(QUIET=True)

        # 加载结构
        wild_structure = parser.get_structure("wild", str(Path(pdb_dir) / wild_name))
        mutant_structure = parser.get_structure("mutant", str(Path(pdb_dir) / mutant_name))

        # 获取并验证原始特征
        wild_orig_features = wild_data['node_features']
        mutant_orig_features = mutant_data['node_features']

        # 验证原始特征维度
        if not validate_features(wild_orig_features, 53, "wild original"):
            return None, None
        if not validate_features(mutant_orig_features, 53, "mutant original"):
            return None, None

        num_residues = len(wild_orig_features)
        logging.info(f"Number of residues: {num_residues}")

        # 创建残基映射
        wild_res_map = {}
        mutant_res_map = {}

        # 处理野生型的残基映射
        for model in wild_structure:
            for residue in model.get_residues():
                res_id = residue.id[1]
                if 0 <= res_id < num_residues:
                    wild_res_map[res_id] = residue

        # 处理突变型的残基映射
        for model in mutant_structure:
            for residue in model.get_residues():
                res_id = residue.id[1]
                if 0 <= res_id < num_residues:
                    mutant_res_map[res_id] = residue

        # 初始化结构特征数组
        wild_struct_features = np.zeros((num_residues, FeatureConfig.STRUCTURE_DIM))
        mutant_struct_features = np.zeros((num_residues, FeatureConfig.STRUCTURE_DIM))

        # 计算野生型结构特征
        for i in range(num_residues):
            if i in wild_res_map:
                try:
                    features = calculate_structure_features_safe(wild_res_map[i], wild_structure)
                    if features is not None and len(features) == FeatureConfig.STRUCTURE_DIM:
                        wild_struct_features[i] = features
                    else:
                        logging.warning(f"Invalid wild structure features for residue {i}")
                except Exception as e:
                    logging.warning(f"Error calculating wild type features for residue {i}: {e}")

        # 计算突变型结构特征
        for i in range(num_residues):
            if i in mutant_res_map:
                try:
                    features = calculate_structure_features_safe(mutant_res_map[i], mutant_structure)
                    if features is not None and len(features) == FeatureConfig.STRUCTURE_DIM:
                        mutant_struct_features[i] = features
                    else:
                        logging.warning(f"Invalid mutant structure features for residue {i}")
                except Exception as e:
                    logging.warning(f"Error calculating mutant features for residue {i}: {e}")

        # 验证结构特征
        if not validate_features(wild_struct_features, FeatureConfig.STRUCTURE_DIM, "wild structure"):
            return None, None
        if not validate_features(mutant_struct_features, FeatureConfig.STRUCTURE_DIM, "mutant structure"):
            return None, None

        # 组合特征
        try:
            wild_combined = np.concatenate([wild_orig_features, wild_struct_features], axis=1)
            mutant_combined = np.concatenate([mutant_orig_features, mutant_struct_features], axis=1)

            # 验证组合特征
            if not validate_features(wild_combined, 67, "wild combined"):
                return None, None
            if not validate_features(mutant_combined, 67, "mutant combined"):
                return None, None

            # logging.info(f"Feature dimensions for {mutant_name}:")
            # logging.info(f"Original features: {wild_orig_features.shape}")
            # logging.info(f"Structure features: {wild_struct_features.shape}")
            # logging.info(f"Combined features: {wild_combined.shape}")

        except Exception as e:
            logging.error(f"Error concatenating features for {mutant_name}: {e}")
            return None, None

        # 查找突变位置
        mutation_pos, mutation_info = find_mutation_position(wild_orig_features, mutant_orig_features)

        if mutation_pos is None:
            logging.error(f"Could not find mutation position for {mutant_name}")
            return None, None

        # 构建数据结构
        data = {
            'wild_type': {
                'node_features': wild_combined,
                'edge_index': wild_data['edge_index'],
                'edge_features': wild_data['edge_features'],
                'mutation_pos': mutation_pos
            },
            'mutant': {
                'node_features': mutant_combined,
                'edge_index': mutant_data['edge_index'],
                'edge_features': mutant_data['edge_features'],
                'mutation_pos': mutation_pos
            },
            'ddg': ddg,
            'wild_type_name': wild_name,
            'mutant_name': mutant_name
        }

        # 构建验证信息
        verification = {
            'file': mutant_name,
            'detected_position': mutation_info['position'],
            'wild_aa': mutation_info['wild_aa'],
            'mutant_aa': mutation_info['mutant_aa'],
            'expected_mutation': mutant_name.split('_mutant_')[1].replace('.pdb', '')
        }

        return data, verification

    except Exception as e:
        logging.error(f"Error processing {mutant_name}: {str(e)}")
        return None, None
def process_dataset(data_path: str, output_path: str, pdb_dir: str, num_workers: int = 4):
    """处理数据集，包含断点续传功能"""
    # 加载原始数据
    with open(data_path, 'rb') as f:
        original_data = pickle.load(f)
        logging.info(f"Loaded {len(original_data)} samples from original dataset")

    # 查找最新的批处理文件
    processed_count, batch_num = find_latest_batch(output_path)
    start_idx = processed_count
    logging.info(f"Resuming from sample {start_idx} (batch {batch_num})")

    # 准备待处理的蛋白质对
    protein_pairs = []
    for item in original_data[start_idx:]:
        args = (
            item['wild_type'],
            item['mutant'],
            item['ddg'],
            item['wild_type_name'],
            item['mutant_name'],
            pdb_dir
        )
        protein_pairs.append(args)
    
    logging.info(f"Will process {len(protein_pairs)} remaining protein pairs")
    
    # 处理计数器
    valid_count = processed_count
    current_batch = []

    # 多进程处理
    with tqdm(total=len(protein_pairs), desc="Processing proteins", ncols=100) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for args in protein_pairs:
                futures.append(executor.submit(process_protein, args))

            for future in as_completed(futures):
                try:
                    result, verification = future.result()
                    if result is not None and verification is not None:
                        if validate_processed_data(result):
                            valid_count += 1
                            current_batch.append(result)

                            # 每50个样本保存一次
                            if len(current_batch) >= 50:
                                batch_num += 1
                                temp_output_path = output_path.replace('.pkl', f'_temp_{valid_count}.pkl')
                                with open(temp_output_path, 'wb') as f:
                                    pickle.dump(current_batch, f)
                                logging.info(f"Saved batch {batch_num} with {len(current_batch)} samples to {temp_output_path}")
                                current_batch = []

                except Exception as e:
                    logging.error(f"Error processing protein: {str(e)}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        'Valid': valid_count,
                        'Batch': batch_num
                    })

    # 保存最后的批次
    if current_batch:
        temp_output_path = output_path.replace('.pkl', f'_temp_{valid_count}.pkl')
        with open(temp_output_path, 'wb') as f:
            pickle.dump(current_batch, f)
        logging.info(f"Saved final batch with {len(current_batch)} samples to {temp_output_path}")

    return valid_count
def validate_processed_data(data: Dict) -> bool:
    """验证处理后的数据"""
    try:
        # 验证野生型数据
        wild = data['wild_type']
        if not validate_features(wild['node_features'], 67, "wild final"):
            return False

        # 验证突变型数据
        mutant = data['mutant']
        if not validate_features(mutant['node_features'], 67, "mutant final"):
            return False

        return True
    except Exception as e:
        logging.error(f"Error validating processed data: {e}")
        return False
def merge_batch_files(output_path: str):
    """合并所有批处理文件"""
    pattern = output_path.replace('.pkl', '_temp_*.pkl')
    batch_files = sorted(glob.glob(pattern), 
                        key=lambda x: int(x.split('_temp_')[-1].replace('.pkl', '')))
    
    if not batch_files:
        logging.error("No batch files found to merge")
        return
    
    merged_data = []
    logging.info(f"Found {len(batch_files)} batch files to merge")
    
    for file in tqdm(batch_files, desc="Merging batch files"):
        try:
            with open(file, 'rb') as f:
                batch_data = pickle.load(f)
                merged_data.extend(batch_data)
            logging.info(f"Merged {file} ({len(batch_data)} samples)")
        except Exception as e:
            logging.error(f"Error merging {file}: {e}")
            continue
    
    # 保存合并后的数据
    if merged_data:
        with open(output_path, 'wb') as f:
            pickle.dump(merged_data, f)
        logging.info(f"Saved {len(merged_data)} total samples to {output_path}")
        
        # 可选：删除临时文件
        # for file in batch_files:
        #     try:
        #         os.remove(file)
        #     except Exception as e:
        #         logging.warning(f"Could not delete {file}: {e}")
    else:
        logging.error("No data to save after merging")

if __name__ == "__main__":
    # 配置参数
    dataset = "350"
    end = ""

    # 设置路径
    base_dir = Path("/media/ST-18T/cheery/CoMDDG-LAPTOP/dataset_process")
    data_path = base_dir / "pkl" / f"data_s{dataset}{end}_add_mutpos_enhanced.pkl"
    output_path = base_dir / "pkl" / f"data_s{dataset}{end}_add_mutpos_enhanced_struct.pkl"
    pdb_dir = base_dir / "Dataset" / f"S{dataset}{end}"

    try:
        # 处理数据集
        total_processed = process_dataset(
            data_path=str(data_path),
            output_path=str(output_path),
            pdb_dir=str(pdb_dir),
            num_workers=20
        )
        
        # 合并所有批处理文件
        merge_batch_files(str(output_path))
        
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        raise
