import logging
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from Bio import PDB
import pickle
import re
import pandas as pd
from tqdm import tqdm

from config import ENCODING_DIM

def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords



aa_dict = {
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


def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords


def positional_encoding(distance, dimension=16):
    encoding = np.zeros(dimension)
    for i in range(dimension):
        encoding[i] = np.sin(distance / (10000 ** ((2 * i) / dimension))) if i % 2 == 0 else np.cos(
            distance / (10000 ** ((2 * i) / dimension)))
    return encoding


def load_conservation_scores(directory, filename):
    conservation_file = os.path.join(directory, filename.replace('.pdb', '_conservation.npy'))
    if os.path.exists(conservation_file):
        return np.load(conservation_file)
    return None


def create_graph_from_structure(structure, pssm_scores, conservation_scores, threshold=8.0, encoding_dim=ENCODING_DIM):
    ca_coords = get_residue_ca_coords(structure)
    graph = nx.Graph()

    # Add nodes with combined features
    for i, ((chain_id, residue_id), coord) in enumerate(ca_coords.items()):
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()
        one_hot = np.array(aa_dict.get(resname, [0] * 20))  # Convert one-hot encoding to a NumPy array
        pssm = np.array(pssm_scores[i]) if pssm_scores is not None and i < len(pssm_scores) else np.zeros(20)
        # 归一化PSSM得分
        pssm = np.clip(pssm, -10, 10)
        pssm = (pssm + 10) / 20
        conservation = np.array([conservation_scores[i]]) if conservation_scores is not None and i < len(
            conservation_scores) else np.zeros(1)
        features = np.concatenate((one_hot, pssm, conservation))  # Concatenate features
        graph.add_node((chain_id, residue_id), feature=features)

    # Add edges based on distance threshold
    for (chain_id1, res_id1), coord1 in ca_coords.items():
        for (chain_id2, res_id2), coord2 in ca_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1), (chain_id2, res_id2), feature=edge_feature)

    return graph


def \
        process_pdb_files(directory, pssm_pickle_file, conservation_directory):
    parser = PDB.PDBParser(QUIET=True)
    graphs = []

    wild_type_files = {}
    mutant_files = {}

    # Load PSSM scores
    with open(pssm_pickle_file, 'rb') as file:
        pssm_scores_dict = pickle.load(file)

    # Classify files as wild-type or mutant
    for file_name in os.listdir(directory):
        if re.match(r'.+_wild_type\.pdb', file_name):
            wild_type_files[file_name.split('_wild_type')[0]] = file_name
        elif re.match(r'.+_mutant_.+\.pdb', file_name):
            base_name = file_name.split('_mutant_')[0]
            if base_name not in mutant_files:
                mutant_files[base_name] = []
            mutant_files[base_name].append(file_name)

    # 获取所有wild type PDB文件
    wild_pdbs = list(Path(directory).glob('*_wild_type.pdb'))
    # 预计算总突变体数量
    total_mutants = sum(len(list(wild_pdb.parent.glob(f'{wild_pdb.stem.replace("_wild_type", "")}_mutant_*.pdb')))
                        for wild_pdb in wild_pdbs)
    # 创建进度条
    with tqdm(total=total_mutants,
              desc="Processing mutations",
              unit="mutation",
              ncols=100,
              colour='green',
              file=sys.stdout) as pbar:
        # Process each mutant file with its corresponding wild type
        for base_name, mutants in mutant_files.items():
            if base_name in wild_type_files:
                wild_type_file = wild_type_files[base_name]
                wild_type_structure = parser.get_structure(base_name + '_wild_type',
                                                           os.path.join(directory, wild_type_file))

                # Load PSSM and conservation scores for wild type
                wild_type_pssm = pssm_scores_dict.get(wild_type_file.replace('.pdb', '.pssm'), None)
                wild_type_conservation = load_conservation_scores(conservation_directory, wild_type_file)
                wild_type_graph = create_graph_from_structure(wild_type_structure, wild_type_pssm, wild_type_conservation)

                for mutant_file in mutants:
                    mutant_structure = parser.get_structure(mutant_file, os.path.join(directory, mutant_file))

                    # Load PSSM and conservation scores for mutant
                    mutant_pssm = pssm_scores_dict.get(mutant_file.replace('.pdb', '.pssm'), None)
                    mutant_conservation = load_conservation_scores(conservation_directory, mutant_file)

                    mutant_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)

                    graphs.append((wild_type_graph, mutant_graph, mutant_file))

                    pbar.update(1)

    return graphs


def process_pdb_files_with_residue_debug(directory, pssm_pickle_file, conservation_directory):
    """
    Enhanced version of process_pdb_files with detailed residue tracking
    """
    from Bio import PDB
    import networkx as nx
    import os
    import logging

    parser = PDB.PDBParser(QUIET=True)
    graphs = []

    def log_residue_info(graph, source):
        """Helper function to log residue information"""
        residue_count = 0
        has_residue_attr = 0
        for node, data in graph.nodes(data=True):
            if 'residue' in data:
                has_residue_attr += 1
                if data['residue'] is not None:
                    residue_count += 1

        # logging.info(f"{source} Graph Stats:")
        # logging.info(f"Total nodes: {len(graph.nodes)}")
        # logging.info(f"Nodes with residue attribute: {has_residue_attr}")
        # logging.info(f"Nodes with valid residue: {residue_count}")

    # Load PSSM scores
    try:
        with open(pssm_pickle_file, 'rb') as file:
            pssm_scores_dict = pickle.load(file)
        logging.info(f"Loaded PSSM scores for {len(pssm_scores_dict)} structures")
    except Exception as e:
        logging.error(f"Error loading PSSM scores: {str(e)}")
        pssm_scores_dict = {}

    # Get wild-type and mutant files
    wild_type_files = {}
    mutant_files = {}

    # Classify files
    for file_name in os.listdir(directory):
        if re.match(r'.+_wild_type\.pdb', file_name):
            wild_type_files[file_name.split('_wild_type')[0]] = file_name
        elif re.match(r'.+_mutant_.+\.pdb', file_name):
            base_name = file_name.split('_mutant_')[0]
            if base_name not in mutant_files:
                mutant_files[base_name] = []
            mutant_files[base_name].append(file_name)

    logging.info(f"Found {len(wild_type_files)} wild type structures")
    logging.info(f"Found {sum(len(muts) for muts in mutant_files.values())} mutant structures")

    with tqdm(total=len(mutant_files), desc="Processing mutations", unit="mutation") as pbar:
        for base_name, mutants in mutant_files.items():
            if base_name in wild_type_files:
                wild_type_file = wild_type_files[base_name]
                try:
                    # Process wild type structure
                    wild_type_struct = parser.get_structure(
                        base_name + '_wild_type',
                        os.path.join(directory, wild_type_file)
                    )

                    # Load scores for wild type
                    wild_type_pssm = pssm_scores_dict.get(wild_type_file.replace('.pdb', '.pssm'), None)
                    wild_type_conservation = load_conservation_scores(conservation_directory, wild_type_file)

                    # Create wild type graph
                    wild_type_graph = create_graph_from_structure_enhanced(
                        wild_type_struct,
                        wild_type_pssm,
                        wild_type_conservation
                    )

                    log_residue_info(wild_type_graph, f"Wild Type {wild_type_file}")

                    for mutant_file in mutants:
                        try:
                            # Process mutant structure
                            mutant_struct = parser.get_structure(
                                mutant_file,
                                os.path.join(directory, mutant_file)
                            )

                            # Load scores for mutant
                            mutant_pssm = pssm_scores_dict.get(mutant_file.replace('.pdb', '.pssm'), None)
                            mutant_conservation = load_conservation_scores(conservation_directory, mutant_file)

                            # Create mutant graph
                            mutant_graph = create_graph_from_structure_enhanced(
                                mutant_struct,
                                mutant_pssm,
                                mutant_conservation
                            )

                            log_residue_info(mutant_graph, f"Mutant {mutant_file}")

                            # Store the pair
                            graphs.append((wild_type_graph, mutant_graph, mutant_file))

                        except Exception as e:
                            logging.error(f"Error processing mutant {mutant_file}: {str(e)}")
                            continue

                except Exception as e:
                    logging.error(f"Error processing wild type {wild_type_file}: {str(e)}")
                    continue

                pbar.update(1)

    return graphs


def create_graph_from_structure_enhanced(structure, pssm_scores=None, conservation_scores=None, threshold=8.0):
    """
    Enhanced version of create_graph_from_structure with better residue handling
    """
    graph = nx.Graph()

    # Track residues and their coordinates
    residue_coords = {}
    residue_info = {}

    # First pass: collect all valid residues and their coordinates
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only process standard amino acid residues
                if residue.get_resname() in aa_dict:
                    if 'CA' in residue:
                        res_id = (chain.id, residue.id)
                        residue_coords[res_id] = residue['CA'].coord
                        residue_info[res_id] = residue
                        logging.debug(f"Found valid residue: {chain.id}:{residue.id} {residue.get_resname()}")

    # Create nodes with residue information
    for i, (res_id, coord) in enumerate(residue_coords.items()):
        chain_id, residue_id = res_id
        residue = residue_info[res_id]
        resname = residue.get_resname()

        # Collect features
        one_hot = np.array(aa_dict.get(resname, [0] * 20))

        # Handle PSSM scores
        pssm = np.zeros(20)
        if pssm_scores is not None and i < len(pssm_scores):
            pssm = np.clip(pssm_scores[i], -10, 10)
            pssm = (pssm + 10) / 20

        # Handle conservation scores
        conservation = np.zeros(1)
        if conservation_scores is not None and i < len(conservation_scores):
            conservation = np.array([conservation_scores[i]])

        # Combine features
        features = np.concatenate((one_hot, pssm, conservation))

        # Add node with both features and residue object
        graph.add_node(res_id, feature=features, residue=residue)

    # Add edges based on distance threshold
    for res_id1, coord1 in residue_coords.items():
        for res_id2, coord2 in residue_coords.items():
            if res_id1 < res_id2:  # Avoid duplicate edges
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, ENCODING_DIM)
                    graph.add_edge(res_id1, res_id2, feature=edge_feature)

    # logging.info(f"Created graph with {len(graph)} nodes and {len(graph.edges)} edges")
    return graph