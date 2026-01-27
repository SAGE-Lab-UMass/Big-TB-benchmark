"""
DNABERT-S Embedding Data Processing and Conversion Pipeline

This script handles the conversion and preprocessing of DNABERT-2 gene embeddings from
various formats (NPY batches) to memory-mapped files for efficient training and inference.

Key Functions:
1. NPY to Memory-Map Conversion: Converts batch NPY files to efficient .mmap format
2. Mean Pooling: Creates compressed embeddings via mean pooling (sequence or dimension)
3. Memory-Efficient Processing: Uses memory mapping to handle large embedding datasets

Supported Operations:
- Convert token-level embeddings from NPY batches to memory-mapped format
- Generate sequence-averaged embeddings (mean_seq): Average over sequence length
- Generate dimension-averaged embeddings (mean_dim): Average over embedding dimension
- Maintain sample identifiers for proper data tracking

File Formats:
- Input: {gene}_tok_chunk_*.npy (token embeddings from DNABERT-2)
- Output: {gene}_tok_chunk_*.mmap + *_meta.npz (memory-mapped with metadata)
- Compressed: {gene}_tok_chunk_*_{mean_method}.mmap (mean-pooled embeddings)

Author: Saishradha Mohanty
"""

# Suppress warnings for cleaner output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import glob
import gc
from pathlib import Path
import argparse
import numpy as np
import tqdm


def npy_to_memmap_with_identifiers(embed_dir, output_base_dir, gene, embed_method, prefix="full"):
    """
    Convert NPY embedding batch files to memory-mapped format for efficient access.
    
    This function processes token-level embeddings stored as NPY batch files and converts
    them to memory-mapped format (.mmap) with corresponding metadata files. This enables
    efficient random access during training without loading entire datasets into RAM.
    
    Args:
        embed_dir (str): Directory containing original NPY embedding files
        output_base_dir (str): Base output directory for memory-mapped files
        gene (str): Gene name (used to locate files like {gene}_tok_chunk_*.npy)
        embed_method (str): Embedding method ("zs" for zero-shot, "ft" for fine-tuned)
        prefix (str): Prefix for sample identifiers (e.g., "full", "train", "val")
    
    Creates:
        - {gene}_tok_chunk_*.mmap: Memory-mapped embedding arrays (float16)
        - {gene}_tok_chunk_*_meta.npz: Metadata files with shape, identifiers, and paths
    
    File Structure:
        Input:  {embed_dir}/{gene}/{embed_method}_{prefix}_embeddings_batch_*.npy
        Output: {output_base_dir}/{gene}/{gene}_tok_chunk_*.mmap + *_meta.npz
    """
    print(f"Converting NPY embeddings to memory-mapped format")
    print(f"  Gene: {gene}")
    print(f"  Method: {embed_method}")
    print(f"  Prefix: {prefix}")
    
    # Construct file search pattern
    src_pattern = os.path.join(embed_dir, gene, f"{embed_method}_{prefix}_embeddings_batch_*.npy")
    output_dir = os.path.join(output_base_dir, gene)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all NPY batch files for this gene
    npy_files = sorted(glob.glob(src_pattern))
    print(f"  Found {len(npy_files)} NPY batch files")

    if not npy_files:
        print(f"  Warning: No files found matching pattern: {src_pattern}")
        return

    # Track sample indexing across batches for consistent identifiers
    global_sample_offset = 0

    # Process each NPY batch file
    for batch_idx, npy_path in enumerate(tqdm.tqdm(npy_files, desc=f"Converting {gene}")):
        # Generate output filenames
        npy_basename = os.path.basename(npy_path)
        mmap_output_path = os.path.join(output_dir, f"{npy_basename}.mmap")
        meta_output_path = os.path.join(output_dir, f"{npy_basename}_meta.npz")

        # Skip if already processed
        if os.path.exists(meta_output_path):
            print(f"Skipping {npy_path} (already processed)")
            continue

        # Load embeddings from NPY file
        # Original shape: (batch_size, 1, sequence_length, embedding_dim)
        # We squeeze the second dimension to get: (batch_size, sequence_length, embedding_dim)
        print(f"    Processing batch {batch_idx + 1}/{len(npy_files)}")
        
        # Load as float32 first (actual token shape is (batch_size, 1, seq_len, 768) -> (batch, seq_len, dim) for DNABERT-2)
        # Convert to float16 for memory efficiency
        embeddings_f16 = np.load(npy_path, mmap_mode="r")[:, 0].astype("float16")   
        batch_shape = embeddings_f16.shape
        
        print(f"Batch shape: {batch_shape}")
        print(f"Data type: {embeddings_f16.dtype}")

        # Create memory-mapped output file
        mmap_array = np.memmap(
            mmap_output_path, 
            mode="w+", 
            dtype="float16", 
            shape=batch_shape
        )
        
        # Copy data to memory-mapped file
        mmap_array[:] = embeddings_f16
        mmap_array.flush()
        
        # Clean up memory
        del mmap_array, embeddings_f16, embeddings

        # Generate sample identifiers for this batch
        batch_size = batch_shape[0]
        batch_identifiers = np.array([
            f"{prefix}_{i:06d}" for i in range(global_sample_offset, global_sample_offset + batch_size)
        ])
        global_sample_offset += batch_size

        # Save metadata file
        np.savez_compressed(
            meta_output_path,
            shape=batch_shape,                          # Array dimensions
            mmap_path=os.path.basename(mmap_output_path),  # Relative path to mmap file
            identifier=batch_identifiers                # Sample identifiers
        )

        print(f"Saved {batch_size} samples with identifiers {batch_identifiers[0]} to {batch_identifiers[-1]}")

    print(f"{gene}: Converted {len(npy_files)} batches to memory-mapped format")
    print(f"  Total samples processed: {global_sample_offset}")
    print(f"  Output directory: {output_dir}")
    
    # Clean up memory
    gc.collect()


def project_gene_to_mean(gene, memmap_dir, mean_dir, mean_method="mean_seq"):
    """
    Create mean-pooled embeddings from token-level memory-mapped files.
    
    This function reduces the dimensionality of token-level embeddings by applying mean pooling either across the sequence length or embedding dimensions. This creates more compact representations suitable for downstream classification tasks.
    
    Args:
        gene (str): Gene name to process
        memmap_dir (str): Directory containing original token-level memory-mapped files
        mean_dir (str): Output directory for mean-pooled embeddings
        mean_method (str): Pooling method, either:
            - "mean_seq": Average over sequence length (seq_len, 768) -> (1, 768)
            - "mean_dim": Average over embedding dimension (seq_len, 768) -> (seq_len, 1)
    
    Creates:
        - {gene}_tok_chunk_*_{mean_method}.mmap: Mean-pooled embeddings
        - {gene}_tok_chunk_*_{mean_method}_meta.npz: Corresponding metadata
    
    Embedding Shape Transformations:
        Original: (batch_size, sequence_length, 768)
        mean_seq: (batch_size, 1, 768) - averaged over sequence positions
        mean_dim: (batch_size, sequence_length, 1) - averaged over embedding dimensions
    """
    print(f"Generating mean-pooled embeddings for gene: {gene}")
    print(f"  Method: {mean_method}")
    print(f"  Input directory: {memmap_dir}")
    print(f"  Output directory: {mean_dir}")
    
    # Validate mean pooling method
    if mean_method not in ["mean_dim", "mean_seq"]:
        raise ValueError(f"mean_method must be 'mean_dim' or 'mean_seq', got '{mean_method}'")
    
    print(f"Projecting {gene} using {mean_method} from {memmap_dir} to {mean_dir}")
    
    # Find all metadata files for this gene
    input_gene_dir = Path(memmap_dir) / gene
    meta_files = sorted(input_gene_dir.glob("*_meta.npz"))

    if not meta_files:
        raise FileNotFoundError(f"No metadata files *_meta.npz found for {gene} in {input_gene_dir}")
    
    print(f"  Found {len(meta_files)} embedding chunks to process")

    # Create output directory
    output_gene_dir = Path(mean_dir) / gene
    output_gene_dir.mkdir(parents=True, exist_ok=True)

    # Process each embedding chunk
    for meta_path in tqdm.tqdm(meta_files, desc=f"Mean pooling {gene}"):
        
        # Load metadata to get shape and identifiers
        meta_path = Path(meta_path)
        meta_data = np.load(meta_path, allow_pickle=True)
        original_shape = tuple(meta_data["shape"])  # (batch_size, seq_len, embedding_dim)
        
        # Generate output file paths
        base_name = meta_path.stem.replace("_meta", "")  # Remove "_meta" suffix
        output_mmap_path = output_gene_dir / f"{base_name}_{mean_method}.mmap"
        output_meta_path = output_gene_dir / f"{base_name}_{mean_method}_meta.npz"
        
        # Skip if already processed
        if output_meta_path.exists() and output_mmap_path.exists():
            continue
        
        print(f"    Processing: {meta_path.name}")
        print(f"      Original shape: {original_shape}")
        
        # Load original memory-mapped embeddings
        input_mmap_path = meta_path.parent / meta_path.name.replace("_meta.npz", ".mmap")
        input_mmap = np.memmap(
            input_mmap_path, 
            dtype="float16", 
            mode="r",
            shape=original_shape
        )

        # Write mean-compressed tensor 
        if mean_method == "mean_dim":
            # Mean over embedding dimension (last axis)
            mm_out = np.memmap(output_mmap_path, dtype="float16", mode="w+", shape=(meta_data["shape"][0], meta_data["shape"][1], 1))
            mm_out[:] = input_mmap.astype("float32").mean(axis=2, keepdims=True).astype("float16")
        elif mean_method == "mean_seq":
            # Mean over sequence length (middle axis)
            mm_out = np.memmap(output_mmap_path, dtype="float16", mode="w+", shape=(meta_data["shape"][0], 1, meta_data["shape"][2]))
            mm_out[:] = input_mmap.astype("float32").mean(axis=1, keepdims=True).astype("float16")
        mm_out.flush()
    
        np.savez(
            output_meta_path,
            identifier=meta_data["identifier"],     # Preserve sample identifiers
            shape=mm_out.shape,                     # New shape after pooling
            mmap_path=output_mmap_path.name         # Relative path to mmap file
        )
        
        print(f"      Saved pooled embeddings: {output_mmap_path.name}")
        
        # Clean up memory
        del mm_out, input_mmap
        gc.collect()
    
    print(f"Mean pooling completed for {gene}")
    print(f"  Method: {mean_method}")
    print(f"  Processed {len(meta_files)} chunks")
    print(f"  Output directory: {output_gene_dir}")


def main(args):
    """
    Main function to orchestrate embedding data processing operations.
    
    This function handles the conversion and preprocessing pipeline based on
    command-line arguments. It can perform either NPY-to-memmap conversion
    or mean pooling operations on gene embeddings.
    
    Args:
        args: Command-line arguments containing processing configuration
    """
    print("=" * 80)
    print("DNABERT-S Embedding Data Processing Pipeline")
    print("=" * 80)
    print(f"Target gene: {args.gene}")
    print(f"Embedding type: {args.embed_type}")
    print(f"Embedding method: {args.embed_method}")
    print("=" * 80)
    
    # Currently configured to run mean pooling operation
    # To run NPY-to-memmap conversion, uncomment the following line:
    # npy_to_memmap_with_identifiers(args.embed_dir, args.memmap_dir, args.gene, args.embed_method, args.embed_name_prefix)
    
    if args.embed_type in ['mean_seq', 'mean_dim']:
        print("\nPerforming mean pooling operation...")
        project_gene_to_mean(
            gene=args.gene, 
            memmap_dir=args.memmap_dir, 
            mean_dir=args.mean_dir, 
            mean_method=args.embed_type
        )
        print("\nMean pooling completed successfully!")
    else:
        print(f"\nWarning: Embedding type '{args.embed_type}' not supported for mean pooling")
        print("Supported types: 'mean_seq', 'mean_dim'")
    
    print("\n" + "=" * 80)
    print("Processing pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embedding memmaps for the model')
    parser.add_argument('--embed_dir', type=str, default='/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/token/', help="Directory having the original embeddings")
    parser.add_argument('--memmap_dir', type=str, default="/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/memmaps", help="Directory to save the memmap embeddings")
    parser.add_argument('--mean_dir', type=str, default='/scratch/workspace/saishradhamo_umass_edu-big-tb/DNABert/embeddings/zero-shot/mean_seq/', help='Directory to save the mean embeddings')

    parser.add_argument('--embed_name_prefix', type=str, default="train", help='Prefix for original embedding names (e.g., "train", "val", "full")')
    parser.add_argument('--gene', type=str, default='inhA', help="Gene name for which embeddings are generated and memmap to be created")
    parser.add_argument('--embed_type', type=str, default='mean_seq', help="Type of embedding to generate (e.g., 'mean_dim', 'mean_seq', 'token')", choices=['mean_dim', 'mean_seq', 'token'])
    parser.add_argument('--embed_method', type=str, default="zs", help='The method used for generating embeddings. Currently support [zs, ft]', choices=["zs", "ft"])


    args = parser.parse_args()
    
    if __name__ == "__main__":
        main(args)
