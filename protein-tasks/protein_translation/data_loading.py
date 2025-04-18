# data_loading.py

import pandas as pd
import os
from evcouplings.align import Alignment
import numpy as np

def load_phenotype_data(paths, drug):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        if drug in df.columns:
            df = df.dropna(subset=[drug])
        else:
            print(f"Warning: {drug} column not found in {path}")
        dfs.append(df)
    phenotype_data = pd.concat(dfs, ignore_index=True)
    print(f"Number of records in phenotype data: {len(phenotype_data)}")
    return phenotype_data

def load_who_catalog(who_catalog_path):
    who_catalog = pd.read_excel(who_catalog_path, sheet_name='Catalogue_master_file', header=2)
    frequency_df = who_catalog[['drug', 'gene', 'mutation', 'variant', 'effect',
                                'Present_R', 'Present_S', 'Present_SOLO_R', 'FINAL CONFIDENCE GRADING']]
    return frequency_df

def load_variants(csv_file_path):
    variants_df = pd.read_csv(csv_file_path)
    return variants_df

def load_alignment(file_path, alphabet='-actg'):
    return Alignment.from_file(open(file_path), alphabet=alphabet)


            
def load_feature_matrix_and_labels(gene_name):
    file_dir ="/work/pi_annagreen_umass_edu/mahbuba/Data-Curation-for-MTB/protein-tasks/data/feature_matrix_labels"
    feature_matrix_file = f'{file_dir}/{gene_name}_feature_matrix.npy'
    print(feature_matrix_file)
    labels_file = f'{file_dir}/{gene_name}_labels.npy'
    
    if os.path.exists(feature_matrix_file) and os.path.exists(labels_file):
        print(f"Loading feature matrix and labels for {gene_name} from disk.")
        feature_matrix = np.load(feature_matrix_file)
        labels = np.load(labels_file)
    else:
        print(f"Need to create feature matrix and labels for {gene_name}.")
        # feature_matrix, labels = create_feature_matrix(subset_alignment, drug, phenotype_data, h37rv_nongap_indices, h37rv_sequence_str, orientation, gene_name, discard)
        
    return feature_matrix, labels                 