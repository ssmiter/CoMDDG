"""dataset_process/enhanced_graph_gen_v2.py"""
from collections import Counter

import pandas as pd
import numpy as np
import re
import pickle
from tqdm import tqdm
from utils import process_pdb_files

from enhanced_feature_gen import calculate_sequence_properties
# S250: PCC=0.90
# from create_data import calculate_sequence_properties
# S250: PCC=0.88，需要判断是否过拟合导致PCC降低了


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


def enhanced_graph_to_gnn_format(graph):
    """增强版的graph转换函数"""
    node_list = list(graph.nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}

    # 获取节点特征并添加序列属性
    nodes = []
    for node, data in graph.nodes(data=True):
        basic_features = data['feature']  # 原有的41维特征
        residue = graph.nodes[node].get('residue')  # 获取残基对象
        if residue:
            seq_props = calculate_sequence_properties(residue)  # 12维序列特征
            combined_features = np.concatenate([basic_features, seq_props])
        else:
            combined_features = np.concatenate([basic_features, np.zeros(12)])
        nodes.append(combined_features)

    nodes = np.array(nodes)
    edges = np.array([(node_index[u], node_index[v]) for u, v in graph.edges]).T
    edge_features = np.array([data['feature'] for _, _, data in graph.edges(data=True)])

    return nodes, edges, edge_features


def save_enhanced_graphs(graphs, csv_file, output_file):
    """保存增强版图数据，使用one-hot编码比较确定突变位点"""
    df = pd.read_csv(csv_file)
    dataset = []
    verification_results = []

    with tqdm(total=len(graphs), desc="Processing protein pairs", ncols=100) as pbar:
        for wild_type_graph, mutant_graph, filename in graphs:
            match = re.match(r'(.+)_([A-Z])_mutant_(.+)\.pdb', filename)
            if match:
                pdb_id, chain, mutation = match.groups()
                ddg_row = df[(df['PDB Id'] == pdb_id) &
                             (df['Mutated Chain'] == chain) &
                             (df['Mutation_PDB'] == mutation)]

                if not ddg_row.empty:
                    try:
                        # 转换图数据
                        wt_nodes, wt_edges, wt_edge_features = \
                            enhanced_graph_to_gnn_format(wild_type_graph)
                        mut_nodes, mut_edges, mut_edge_features = \
                            enhanced_graph_to_gnn_format(mutant_graph)

                        # 通过one-hot比较找到突变位点
                        mutation_pos, mutation_info = find_mutation_position(wt_nodes, mut_nodes)

                        if mutation_pos is not None:
                            ddg = ddg_row['DDGexp'].values[0]
                            mutation_info['expected_mutation'] = mutation  # 保存预期的突变信息

                            data = {
                                'wild_type': {
                                    'node_features': wt_nodes,
                                    'edge_index': wt_edges,
                                    'edge_features': wt_edge_features,
                                    'mutation_pos': mutation_pos
                                },
                                'mutant': {
                                    'node_features': mut_nodes,
                                    'edge_index': mut_edges,
                                    'edge_features': mut_edge_features,
                                    'mutation_pos': mutation_pos
                                },
                                'ddg': ddg,
                                'wild_type_name': f"{pdb_id}_{chain}_wild_type.pdb",
                                'mutant_name': filename
                            }
                            dataset.append(data)

                            # 添加验证信息
                            verification_results.append({
                                'file': filename,
                                'detected_position': mutation_info['position'],
                                'wild_aa': mutation_info['wild_aa'],
                                'mutant_aa': mutation_info['mutant_aa'],
                                'expected_mutation': mutation_info['expected_mutation']
                            })

                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue

                    pbar.set_postfix({
                        'Current': f"{pdb_id}_{chain}",
                        'Saved': len(dataset)
                    })
            pbar.update(1)

    print(f"\nTotal processed: {len(dataset)} pairs")

    # 保存验证结果
    verification_file = output_file.replace('.pkl', '_verification.pkl')
    with open(verification_file, 'wb') as f:
        pickle.dump(verification_results, f)
    print(f"Verification results saved to {verification_file}")

    # 保存处理后的数据集
    if dataset:
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {output_file}")
    else:
        print("No valid samples to save")

def process_pdb_files_with_residue(directory, pssm_pickle_file, conservation_directory):
    """修改原有的process_pdb_files函数以保存残基信息"""
    graphs = process_pdb_files(directory, pssm_pickle_file, conservation_directory)

    # 为每个图添加残基信息
    enhanced_graphs = []
    for wild_graph, mutant_graph, filename in graphs:
        # 为野生型图的节点添加残基信息
        for node in wild_graph.nodes():
            residue = wild_graph.nodes[node].get('residue')
            if residue:
                wild_graph.nodes[node]['residue'] = residue

        # 为突变型图的节点添加残基信息
        for node in mutant_graph.nodes():
            residue = mutant_graph.nodes[node].get('residue')
            if residue:
                mutant_graph.nodes[node]['residue'] = residue

        enhanced_graphs.append((wild_graph, mutant_graph, filename))

    return enhanced_graphs


def verify_dataset_files(dataset_file, verification_file):
    # 加载文件
    print("Loading files...")
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    with open(verification_file, 'rb') as f:
        verification = pickle.load(f)

    print("\nBasic Statistics:")
    print(f"Number of samples in dataset: {len(dataset)}")
    print(f"Number of verification records: {len(verification)}")

    # 检查数据集内容
    print("\nDataset Content Check:")
    sample = dataset[0]
    print("\nFeature dimensions:")
    print(f"Wild-type node features: {sample['wild_type']['node_features'].shape}")
    print(f"Wild-type edge features: {sample['wild_type']['edge_features'].shape}")
    print(f"Mutant node features: {sample['mutant']['node_features'].shape}")
    print(f"Mutant edge features: {sample['mutant']['edge_features'].shape}")

    # 检查突变位置的分布
    mutation_positions = [data['wild_type']['mutation_pos'] for data in dataset]
    print("\nMutation Position Statistics:")
    print(f"Min position: {min(mutation_positions)}")
    print(f"Max position: {max(mutation_positions)}")
    print(f"Average position: {np.mean(mutation_positions):.2f}")

    # 检查验证信息
    print("\nVerification Content Check:")
    mutation_types = [(v['wild_aa'], v['mutant_aa']) for v in verification]
    mutation_counts = Counter(mutation_types)
    print(f"\nTop 5 most common mutations:")
    for (wild, mutant), count in mutation_counts.most_common(5):
        print(f"Wild-type AA {wild} to Mutant AA {mutant}: {count} times")

    # 检查一致性
    print("\nConsistency Check:")
    matched_files = sum(1 for d, v in zip(dataset, verification)
                        if d['mutant_name'] == v['file'])
    print(f"Files matched between dataset and verification: {matched_files}")

    # 验证特征维度
    feature_dims = set(data['wild_type']['node_features'].shape[1] for data in dataset)
    print(f"\nUnique node feature dimensions: {feature_dims}")

    return True

if __name__ == "__main__":
    # dataset = "2648"  # 或其他数据集名称
    # end = "_NEW"     # 或其他后缀
    dataset = "2648"  # 或其他数据集名称
    end = "_ProtDDG"     # 或其他后缀
    # 处理数据
    directory = f'./Dataset/S{dataset}{end}'
    pssm_pickle_file = f'./Dataset/PSSM_{dataset}{end}/pssm_s{dataset}{end}.pkl'
    conservation_directory = f'./Dataset/cons_s{dataset}{end}'
    csv_file = f'./Dataset/S{dataset}{end}.csv'
    output_file = f'./pkl/data_s{dataset}{end}_add_mutpos_enhanced.pkl'

    # 使用修改后的函数处理PDB文件
    graphs = process_pdb_files_with_residue(directory, pssm_pickle_file, conservation_directory)

    # 保存增强版的图数据
    save_enhanced_graphs(graphs, csv_file, output_file)
    print(f"Processing completed. Results saved to {output_file}")

    # 验证
    dataset_file = output_file
    verification_file = f'./pkl/data_s{dataset}{end}_add_mutpos_enhanced_verification.pkl'

    verify_dataset_files(dataset_file, verification_file)