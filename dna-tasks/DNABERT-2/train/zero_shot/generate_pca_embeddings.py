"""
PCA Dimensionality Reduction for DNABERT-2 Gene Embeddings

This script performs Principal Component Analysis (PCA) on high-dimensional gene embeddings
to reduce memory usage and computational complexity for downstream resistance classification.

The process involves:
1. Fitting a joint PCA across all genes for each drug (to capture cross-gene relationships)
2. Projecting individual gene embeddings into the reduced PCA space
3. Saving the compressed embeddings for efficient loading during training

Key benefits:
- Reduces embedding dimensionality from 768 to k_components (e.g., 10)
- Maintains most important variance while reducing memory footprint
- Enables faster training and inference

Author: Saishradha Mohanty
"""

import numpy as np
import glob
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


# Drug-to-gene mappings for resistance prediction
# These define which genes are analyzed for each drug's resistance mechanisms
SINGLE_GENE_DRUGS = {
    "PYRAZINAMIDE": ["pncA"],  # Pyrazinamidase gene
    "AMIKACIN": ["rrs"],       # 16S rRNA gene 
    "KANAMYCIN": ["rrs"],      # 16S rRNA gene
}

MULTI_GENE_DRUGS = {
    "RIFAMPICIN": ["rpoB", "rpoC"],           # RNA polymerase subunits
    "CAPREOMYCIN": ["tlyA", "rrs", "rrl"],    # Multiple ribosomal targets
    "STREPTOMYCIN": ["rpsL", "rrs", "gid"],   # Ribosomal proteins and rRNA
    "ISONIAZID": ["katG", "inhA"],            # Catalase and enoyl reductase
    "ETHIONAMIDE": ["ethA", "ethR"],          # Ethionamide activation genes
    "MOXIFLOXACIN": ["gyrB", "gyrA"],         # DNA gyrase subunits
    "LEVOFLOXACIN": ["gyrB", "gyrA"],         # DNA gyrase subunits
    "ETHAMBUTOL": ["embC", "embA", "embB"],   # Arabinosyl transferases
}

# Combine all drug mappings
ALL_DRUGS = {**SINGLE_GENE_DRUGS, **MULTI_GENE_DRUGS}

# Currently processing drug (modify this to run specific drugs)
CURRENT_DRUG = {"STREPTOMYCIN": ["rpsL", "gid", "rrs"]}


def fit_joint_pca(genes, k_components, batch_size=10000, 
                  root_dir="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps", 
                  overwrite=False):
    """
    Fit a joint PCA model across all genes for a drug to capture cross-gene relationships.
    
    This function uses IncrementalPCA to process large embedding files without loading
    everything into memory at once. The PCA is fitted on the concatenated embedding space
    of all genes associated with a drug.
    
    Args:
        genes (list): List of gene names (e.g., ['katG', 'inhA'] for ISONIAZID)
        k_components (int): Number of principal components to keep
        batch_size (int): Number of sequence embeddings to process per batch
        root_dir (str): Base directory containing gene embedding memmap files
        overwrite (bool): Whether to overwrite existing PCA files
    
    Returns:
        Path: Path to the saved PCA components file (.npz format) in a format:
              {root_dir}/{gene1}_{gene2}_..._pc{k_components}.npz
        
    Output file contains:
        - mean: Global mean vector for centering (shape: 768,) where 768 is embedding dimension for DNABERT-2
        - components: Principal component vectors (shape: k_components, 768)
    """
    # Create descriptive filename from gene names
    gene_tag = "_".join(genes)  # e.g., katG_inhA
    pca_file_path = Path(root_dir) / f"{gene_tag}_pc{k_components}.npz"  # e.g., katG_inhA_pc10.npz
    
    print(f"PCA fitting for genes: {genes}")
    print(f"Output file: {pca_file_path}")
    
    # Skip if already computed (unless overwrite requested)
    if pca_file_path.exists() and not overwrite:
        print(f"PCA already exists → {pca_file_path}")
        return pca_file_path

    # Initialize Incremental PCA for memory-efficient processing
    # IncrementalPCA processes data in batches instead of forming one big (ΣL x 768) matrix in RAM , maintaining running statistics
    ipca = IncrementalPCA(n_components=k_components, batch_size=batch_size)

    print(f"Fitting PCA with {k_components} components...")
    
    # Process each gene's embedding files
    for gene in genes:
        print(f"  Processing gene: {gene}")
        
        # Find all memmap metadata files for this gene
        meta_file_pattern = f"{root_dir}/{gene}/*_meta.npz"
        meta_paths = glob.glob(meta_file_pattern)
        
        if not meta_paths:
            print(f"    Warning: No embedding files found for {gene} at {meta_file_pattern}")
            continue
            
        # Process each embedding chunk file
        for meta_path in tqdm(meta_paths, desc=f"PCA fit {gene}"):
            # Load metadata to get memmap shape information
            meta = np.load(meta_path, allow_pickle=True)  
            
            # Load corresponding memmap file
            mmap_path = meta_path.replace("_meta.npz", ".mmap")
            mmap_array = np.memmap(
                mmap_path,
                dtype="float16", 
                mode="r", 
                shape=tuple(meta["shape"])  # (num_isolates, sequence_length, 768)
            )
            
            # Process each isolate's sequence embedding
            for isolate_embedding in mmap_array:  # Shape: (sequence_length, 768)
                # Convert to float32 (required by IncrementalPCA)
                # Each call to partial_fit updates the running PCA statistics
                ipca.partial_fit(isolate_embedding.astype("float32"))

    # Save the fitted PCA components and mean
    np.savez(
        pca_file_path,
        mean=ipca.mean_,           # Global mean for centering (768,)
        components=ipca.components_ # Principal components (k_components, 768)
    )
    
    print(f"Joint PCA saved to: {pca_file_path}")
    print(f"  Explained variance ratio: {ipca.explained_variance_ratio_[:5]}...")  # First 5 components
    
    return pca_file_path


def project_gene_to_pca(gene, pca_components_file, k_components,
                       root_dir="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/train/new/memmaps"):
    """
    Project a gene's embeddings into the PCA space defined by the components file.
    
    This function takes the fitted PCA components and applies the transformation to
    each gene's embedding files, creating compressed versions for efficient loading.
    
    Args:
        gene (str): Gene name (e.g., 'katG')
        pca_components_file (str/Path): Path to PCA components .npz file
        k_components (int): Number of PCA components
        root_dir (str): Base directory containing gene embedding files
    
    Creates:
        - PCA-transformed memmap files in {root_dir}/PCA/{gene}/ directory
        - Corresponding metadata files for each transformed embedding chunk
    
    File naming:
        Original: {gene}_tok_chunk_0.mmap + {gene}_tok_chunk_0_meta.npz
        PCA:      {gene}_tok_chunk_0_pc{k}_meta.npz + {gene}_tok_chunk_0_pc{k}.mmap
    """
    print(f"Projecting gene {gene} embeddings to PCA space...")
    
    # Load PCA transformation parameters
    pca_data = np.load(pca_components_file)
    components, mean = pca_data["components"], pca_data["mean"]  # (k, 768), (768,)
    
    print(f"  PCA components shape: {components.shape}")
    print(f"  PCA mean shape: {mean.shape}")

    # Find all memmap metadata files for this gene
    meta_file_pattern = f"{root_dir}/{gene}/*_meta.npz"
    meta_paths = sorted(Path(root_dir).glob(f"{gene}/*_meta.npz"))
    
    if not meta_paths:
        print(f"  Warning: No embedding files found for {gene}")
        return

    # Create PCA output directory
    pca_output_dir = Path(f"{root_dir}/PCA/{gene}")
    pca_output_dir.mkdir(exist_ok=True, parents=True)
    print(f"  Output directory: {pca_output_dir}")

    # Process each embedding chunk file
    for meta_path in tqdm(meta_paths, desc=f"Projecting {gene}"):
        
        # Generate output filenames
        base_name = meta_path.stem.replace("_meta", "")  # e.g., "katG_tok_chunk_0"
        output_meta_path = pca_output_dir / f"{base_name}_pc{k_components}_meta.npz"
        output_mmap_path = pca_output_dir / f"{base_name}_pc{k_components}.mmap"

        # Skip if both output files already exist
        if output_meta_path.exists() and output_mmap_path.exists():
            continue
            
        # Load source metadata and memmap
        source_meta = np.load(meta_path, allow_pickle=True)
        source_mmap_path = str(meta_path).replace("_meta.npz", ".mmap")
        source_mmap = np.memmap(
            source_mmap_path, 
            dtype="float16", 
            mode="r",
            shape=tuple(source_meta["shape"])  # (num_isolates, seq_len, 768)
        )
        
        # Create output memmap with reduced dimensions
        output_shape = (source_meta["shape"][0], source_meta["shape"][1], k_components)
        output_mmap = np.memmap(
            output_mmap_path, 
            dtype="float16", 
            mode="w+",
            shape=output_shape  # (num_isolates, seq_len, k_components)
        )

        # Transform each isolate's embedding
        for i in range(source_meta["shape"][0]):  # Loop over isolates
            # Load isolate embedding: (seq_len, 768)
            isolate_embedding = source_mmap[i].astype("float32")
            
            # Apply PCA transformation: center then project
            centered_embedding = isolate_embedding - mean  # (seq_len, 768)
            transformed_embedding = centered_embedding @ components.T  # (seq_len, k_components)
            
            # Store as float16 to save space
            output_mmap[i] = transformed_embedding.astype("float16")

        # Ensure data is written to disk
        output_mmap.flush()
        
        # Save metadata for the transformed file
        np.savez_compressed(
            output_meta_path,
            identifier=source_meta["identifier"],  # Preserve sample identifiers
            shape=output_mmap.shape,               # New shape with reduced dimensions
            mmap_path=output_mmap_path.name        # Relative path to mmap file
        )

    print(f"Completed projection for {gene}")


def main():
    """
    Main function to run PCA fitting and projection for specified drugs.
    
    This function orchestrates the entire PCA process:
    1. For each drug, fit a joint PCA across all its associated genes
    2. Project each gene's embeddings using the fitted PCA
    3. Save the compressed embeddings for downstream training
    """
    # Configuration
    K_COMPONENTS = 10  # Number of principal components to retain
    
    print("=" * 80)
    print("DNABERT-S Embedding PCA Compression")
    print(f"Target components: {K_COMPONENTS}")
    print("=" * 80)
    
    # Process each drug in the current batch
    for drug_name, gene_list in CURRENT_DRUG.items():
        print(f"\nProcessing drug: {drug_name}")
        print(f"Associated genes: {gene_list}")
        print("-" * 50)
        
        # Step 1: Fit joint PCA across all genes for this drug
        # This captures cross-gene relationships and shared variance
        pca_components_file = fit_joint_pca(gene_list, K_COMPONENTS)
        
        # Step 2: Project each gene's embeddings into the PCA space
        for gene in gene_list:
            project_gene_to_pca(gene, pca_components_file, K_COMPONENTS)
        
        print(f"\n {drug_name}: PCA processing complete")
        print(f"  Components file: {pca_components_file}")
        print(f"  Compressed embeddings saved for genes: {gene_list}")
    
    print("\n" + "=" * 80)
    print("PCA compression completed successfully!")
    print("Compressed embeddings are ready for efficient training.")
    print("=" * 80)


if __name__ == "__main__":
    main()