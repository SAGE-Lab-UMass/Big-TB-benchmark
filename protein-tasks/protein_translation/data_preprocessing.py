# data_processing.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler

import random


def drop_identical_columns(X):
    X_T = X.T
    _, unique_indices = np.unique(X_T, axis=0, return_index=True)
    unique_indices_sorted = np.sort(unique_indices)
    X_unique = X_T[unique_indices_sorted].T
    return X_unique, unique_indices_sorted

def drop_identical_sequences(X):
    _, unique_indices = np.unique(X, axis=0, return_index=True)
    X_unique = X[unique_indices]
    return X_unique, unique_indices


def encode_labels(labels):
    return [1 if label=="R" else 0 for label in labels]

def filter_nan_labels(labels, *arrays):
    valid_indices = [i for i, label in enumerate(labels) if label != 'nan' and label !='I']
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_arrays = [[array[i] for i in valid_indices] for array in arrays]
    return filtered_labels, filtered_arrays

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# def select_subset_alignment(alignment, start, end, reference_numbers):
#     start_index = np.where(reference_numbers == start)[0][0]
#     end_index = np.where(reference_numbers == end)[0][0]
#     selection = np.arange(start_index, end_index)
#     print("selected columns", selection)
#     return alignment.select(columns=selection)

def select_subset_alignment(alignment, start, end, reference_numbers):
    try:
        # Find closest valid indices within the reference numbers
        start_index_candidates = np.where(reference_numbers == start)[0]
        end_index_candidates = np.where(reference_numbers == end)[0]

        if len(start_index_candidates) == 0:
            print(f"Warning: Start index {start} not found in reference numbers. Using minimum available index.")
            start_index = 0  # Set to minimum valid index
        else:
            start_index = start_index_candidates[0]

        if len(end_index_candidates) == 0:
            print(f"Warning: End index {end} not found in reference numbers. Using maximum available index.")
            end_index = alignment.matrix.shape[1] - 1  # Set to max valid index
        else:
            end_index = end_index_candidates[0]

        # Ensure start_index and end_index are within valid range
        start_index = max(0, min(start_index, alignment.matrix.shape[1] - 1))
        end_index = max(0, min(end_index, alignment.matrix.shape[1]))

        # Ensure start is less than end
        if start_index >= end_index:
            print(f"Error: Invalid range (start_index={start_index}, end_index={end_index}). Returning None.")
            return None

        selection = np.arange(start_index, end_index)
        # print("Selected columns:", selection)

        return alignment.select(columns=selection)
    except IndexError as e:
        print(f"Error in select_subset_alignment: {e}")
        return None


def find_closest_index(reference_numbers, target_index):
    # Check if the exact index exists
    result = np.where(reference_numbers == target_index)
    # Check if we found the gene_start value in the matrix
    if result[0].size > 0:
        # If found, return the first occurrence's column index
        return target_index
    else: 
        # If not, look for the closest index within a range
        # can adjust the range as needed; here it's set to 10
        for i in range(1, 11):
            # Check both one less and one more than the target index
            lower_index = target_index - i
            upper_index = target_index + i
            
            # Check if the lower index exists
            if(np.where(reference_numbers == lower_index)[0].size):
                return lower_index
            # Check if the upper index exists   
            elif(np.where(reference_numbers == upper_index)[0].size):
                return upper_index
            else:
                continue

        # If no match is found within the range, return None or raise an error
        return None

def find_column_for_gene(reference_numbers, gene_start):
    """
    Find the column in the h37rv_numbers that contains the gene_start.
    """
    # Look for the gene_start value in the h37rv_numbers matrix
    result = np.where(reference_numbers == gene_start)
    
    # Check if we found the gene_start value in the matrix
    if result[0].size > 0:
        # If found, return the first occurrence's column index
        return result[1][0]
    else:
        # If not found, return None or raise an error
        return None

def sort_gene_indices(reference_numbers, start_index, end_index,alignment):
    closest_index = find_closest_index(reference_numbers, start_index)
    # if closest_index is not None:
    #     print(f"Closest index to {start_index} found at position {closest_index} in h37rv_numbers.")
    # else:
    #     print(f"No close index to {start_index} found in h37rv_numbers.")
    start_index = closest_index
    closest_index = find_closest_index(reference_numbers, end_index)
    # if closest_index is not None:
    #     print(f"Closest index to {end_index} found at position {closest_index} in h37rv_numbers.")
    # else:
    #     print(f"No close index to {end_index} found in h37rv_numbers.")
    end_index = closest_index  
        


    # Find the column index for the current gene's start index
    column_index = find_column_for_gene(reference_numbers, start_index)
    # Proceed only if the column index was found
    if column_index is not None:
        # Select the subset alignment for the current gene using the found column index
        subset_alignment = select_subset_alignment(alignment, start_index, end_index, reference_numbers[:, column_index])
        # Process the subset_alignment as needed
        print(f"Processed subset alignment for gene start {start_index} and end {end_index} in column {column_index}")
    else:
        # Handle the case where the start index wasn't found in any column
        print(f"Gene start index {start_index} not found in h37rv_numbers.")

    return subset_alignment, column_index, start_index, end_index


def isolate_sequences_with_phenotype(alignments, phenotype_data, filenames):
    isolates_included = set(phenotype_data.Isolate_mapped)
    seqs_to_select = np.array([y for y, x in enumerate(filenames) if x in isolates_included])
    filtered_alignment = alignments.select(sequences=seqs_to_select)
    return filtered_alignment
        




def convert_to_onehot_with_reference(aa_seq, ref_aa):
    # return np.array([1 if aa == ref else 0 for aa, ref in zip(aa_seq, ref_aa)])
    return np.array([0 if aa == ref else 1 for aa, ref in zip(aa_seq, ref_aa)])

def encode_sequence(sequence, reference_length, h37rv_aa_str):
    encoded = convert_to_onehot_with_reference(str(sequence), str(h37rv_aa_str))
    return encoded

def get_aa_positions_by_gene(df, gene_name):
    # Filter the dataframe by the specified gene name
    gene_df = df[df['gene'] == gene_name]
    
    # Extract and sort unique `aa_pos` values
    aa_positions = sorted(gene_df['aa_pos'].unique())
    
    return aa_positions




def seed_everything(seed=42):
    """
    Set a common random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
