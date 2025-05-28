import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import vcf
import itertools
import argparse

from utils import merge_entries

# Load global variables
h37Rv_genes_df = pd.read_csv("data/genbank_reference_data/H37Rv/mycobrowser_h37rv_genes_v4.csv")
h37Rv_coords_to_gene = pd.read_csv("data/genbank_reference_data/H37Rv/h37Rv_coords_to_gene.csv")
h37Rv_coords_to_gene_dict = dict(zip(h37Rv_coords_to_gene['pos'], h37Rv_coords_to_gene['region']))
h37Rv = SeqIO.read("data/genbank_reference_data/H37Rv/GCF_000195955.2_ASM19595v2_genomic.gbff", "genbank")

def check_is_snp(record):
    """
    Check if a variant is a SNP (not an indel or imprecise variant).

    Parameters:
        record (vcf.model._Record): The VCF record to check.

    Returns:
        bool: True if the variant is a SNP, False otherwise.
    """

    if 'IMPRECISE' in record.INFO.keys() or 'SVTYPE' in record.INFO.keys():
        return False
    
    alt_allele = "".join(np.array(record.ALT).astype(str))
    return len(record.REF) == len(alt_allele) and record.INFO['IC'] == 0 and record.INFO['DC'] == 0

def get_codon_from_pos(pos, h37Rv_genes_df):
    """
    Returns a list of H37Rv coordinates in the same codon as the provided position. Sense doesn't matter in this script because all variants are stored in the positive sense direction.

    Parameters:
        pos (int): Position of the variant.
        h37Rv_genes_df (pd.DataFrame): DataFrame containing gene information.

    Returns:
        np.ndarray: Array of unique codon coordinates.
    """
    codon_lst = []
    gene = h37Rv_coords_to_gene_dict[pos]

    # if it starts with NC -> not a gene
    if gene.startswith('NC'):
        return None
    
    # multiple genes because they overlap
    gene = gene.split(',') if ',' in gene else [gene]
    
    # iterate through to account for multiple genes because they overlap
    for single_gene in gene:
        if single_gene not in h37Rv_genes_df.Symbol.values:
            raise ValueError(f"{single_gene} not found in the reference genbank dataset.")
        
        start, end = h37Rv_genes_df.query("Symbol==@single_gene")[['Start', 'End']].values[0]

        # Extract codon coordinates based on reading frame (considering the strand)
        for k in range(start, end, 3):
            # return the codon in which the argument position is in
            if pos in [k, k+1, k+2]:
                codon_lst.append([k, k+1, k+2])

    return np.sort(np.unique(list(itertools.chain.from_iterable(codon_lst))))

def get_variants_within_codon_to_fix(vcf_file):
    """
    Identifies variant positions within the same codon that need to be fixed.

    Parameters:
        vcf_file (str): Path to the VCF file.

    Returns:
        np.ndarray: Array of unique variant positions that require fixing.
    """
    # Initialize list to store positions that need fixing
    pos_to_fix = [] 
    vcf_reader = vcf.Reader(filename=vcf_file)

    # Load all records from the VCF file into a list
    all_records = list(vcf_reader)

    # Iterate through each record and compare with subsequent records
    for i, record in enumerate(all_records[:-1]):
        # Get codon coordinates for the current variant position
        codon_coords = get_codon_from_pos(record.POS, h37Rv_genes_df)

        # Skip non-coding regions
        if codon_coords is None:
            continue

        # Check subsequent records for overlapping codons
        for next_record in all_records[i+1:]:
            if next_record.POS in codon_coords:
                # Ensure both variants are SNPs
                if check_is_snp(record) and check_is_snp(next_record):
                    # Record the position for fixing
                    pos_to_fix.append(record.POS)
                    break  # No need to check further for this record

    # Return unique positions as a sorted array
    return np.unique(pos_to_fix)

def combine_all_variants_single_codon(pos, vcf_fName):
    """
    Combines all variants in the same codon into a single VCF entry.

    Parameters:
        pos (int): Position of the variant in the codon.
        vcf_fName (str): Path to the VCF file.

    Returns:
        combined_entry (vcf.model._Record): The combined VCF entry.
        coords_of_codon (list): List of all coordinates in the codon.
    """
    combined_entry = None
    vcf_reader = vcf.Reader(filename=vcf_fName)

    # get all coordinate in the codon (or multiple codons, in the case of overlapping genes) to check for variants
    # for overlapping genes, the combined variant may be longer, spanning 6 nucleotides
    coords_of_codon = get_codon_from_pos(pos, h37Rv_genes_df)

    for record in vcf_reader:
        if record.POS in coords_of_codon:
            # Combine information from multiple variants into a single entry
            if combined_entry is None:
                combined_entry = record
            else:
                combined_entry = merge_entries(combined_entry, record, coords_of_codon, h37Rv)

    return combined_entry, coords_of_codon

def process_vcf_file(vcf_file, output_file):
    """
    Processes a VCF file and combines variants on the same codon.
    """
    pos_to_fix = get_variants_within_codon_to_fix(vcf_file)
    combined_records, coords_full_lst = [], []
    for pos in pos_to_fix:
        combined_record, codon_coords = combine_all_variants_single_codon(pos, vcf_file)
        combined_records.append(combined_record)
        coords_full_lst.append(codon_coords)
    coords_full_lst = np.unique(list(itertools.chain.from_iterable(coords_full_lst)))
    unchanged_records = []
    vcf_reader = vcf.Reader(filename=vcf_file)

    for record in vcf_reader:
        if record.POS not in coords_full_lst:
            unchanged_records.append(record)
    all_records = sorted(unchanged_records + combined_records, key=lambda r: r.POS)
    vcf_writer = vcf.Writer(open(output_file, "w"), vcf_reader)
    for record in all_records:
        vcf_writer.write_record(record)
    vcf_writer.close()

def process_vcf_files_from_list(file_list_path):
    """
    Processes multiple VCF files listed in a text file.
    """
    with open(file_list_path, "r") as f:
        vcf_files = [line.strip() for line in f.readlines()]
    for vcf_file in vcf_files:
        output_file = vcf_file.replace(".vcf", "_combinedCodons.vcf")
        print(f"Processing {vcf_file} -> {output_file}")
        process_vcf_file(vcf_file, output_file)

def main():
    """
    Main function to parse arguments and process VCF files.
    """
    # TODO: Remove the dupicate rows in the output file
    parser = argparse.ArgumentParser(description="Combine SNPs in VCF files that occur on the same codon.")
    parser.add_argument("-i", "--input", required=True, help="Path to input VCF file or a text file with a list of VCF file paths.")
    parser.add_argument("-t", "--is_text_file", action="store_true", help="Indicates the input is a text file with a list of VCF files or a single VCF file.")
    args = parser.parse_args()

    if args.is_text_file:
        process_vcf_files_from_list(args.input)
    else:
        output_file = args.input.replace(".vcf", "_combinedCodons.vcf")
        print(f"Processing {args.input} -> {output_file}")
        process_vcf_file(args.input, output_file)

if __name__ == "__main__":
    main()
