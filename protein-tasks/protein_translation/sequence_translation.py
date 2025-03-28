from Bio.Seq import Seq
import re
from Bio.Data import CodonTable
import numpy as np


def align_and_handle_deletions(translated_seq, ref_seq):
    """Handle deletions and ensure proper alignment with placeholders for gaps."""
    aligned_seq = []
    ref_index = 0
    trans_index = 0

    while ref_index < len(ref_seq) and trans_index < len(translated_seq):
        if translated_seq[trans_index] == ref_seq[ref_index]:
            # If they match, add the amino acid from the translated sequence
            aligned_seq.append(translated_seq[trans_index])
            trans_index += 1
        else:
            if translated_seq[trans_index] == '-':
                # If there is a gap in the translated sequence, mark it as a gap
                aligned_seq.append('-')
                trans_index += 1
            elif ref_seq[ref_index] == '-':
                # If the reference has a gap, skip the reference gap
                ref_index += 1
            else:
                # If it's a substitution (mismatch), append the amino acid from the translated sequence
                aligned_seq.append(translated_seq[trans_index])
                trans_index += 1

        # Move the reference index forward regardless
        ref_index += 1

    # Ensure the sequence is truncated to the reference length
    aligned_seq = aligned_seq[:len(ref_seq)]

    # If translated sequence is shorter, pad with gaps at the end
    while len(aligned_seq) < len(ref_seq):
        aligned_seq.append('-')

    return ''.join(aligned_seq)

def map_dna_gaps_to_protein_gaps(gap_indices, dna_seq_length):
    """Map DNA gap indices to protein gap indices."""
    protein_gap_indices = []
    for gap in gap_indices:
        # Each protein corresponds to 3 DNA bases, so divide by 3 to map
        protein_gap = gap // 3
        if protein_gap < dna_seq_length // 3:
            protein_gap_indices.append(protein_gap)
    return protein_gap_indices



def translate_sequence_with_gaps(dna_seq, table="Standard", to_stop=False, handle_stops='remove', ref_protein_length=None):
    codon_table = CodonTable.unambiguous_dna_by_name[table]
    standard_table = codon_table.forward_table
    stop_codons = codon_table.stop_codons

    dna_seq = dna_seq.upper()
    protein_seq = []
    frameshift_mutations = False

    seq_len = len(dna_seq)
    i = 0
    cumulative_gap_count = 0

    while i + 3 <= seq_len:
        codon = dna_seq[i:i+3]
        if '-' in codon:
            # Codon contains gap(s)
            cumulative_gap_count += codon.count('-')
            protein_seq.append('-')
        elif re.search(r'[^ATCG]', codon):
            # Codon contains ambiguous nucleotide(s)
            protein_seq.append('X')
        else:
            if codon in stop_codons:
                if to_stop:
                    break
                else:
                    if handle_stops == 'remove':
                        pass  # Do not add any symbol
                    elif handle_stops == 'replace':
                        protein_seq.append('X')
                    else:
                        protein_seq.append('*')
            else:
                amino_acid = standard_table.get(codon)
                if amino_acid:
                    protein_seq.append(amino_acid)
                else:
                    protein_seq.append('X')
        i += 3

    # Handle remaining nucleotides at the end
    if i < seq_len:
        remaining = dna_seq[i:]
        if '-' in remaining or re.search(r'[^ATCG]', remaining):
            protein_seq.append('-')
        else:
            # Remaining nucleotides less than a codon length
            # Do not flag as frameshift
            pass

    # Detect frameshift mutations
    # Compare the length of the translated protein to the reference protein
    if ref_protein_length is not None:
        translated_length = len(protein_seq)
        if translated_length != ref_protein_length:
            frameshift_mutations = True

    # Detect internal stop codons
    if '*' in protein_seq[:-1]:  # Exclude the last amino acid
        frameshift_mutations = True

    return ''.join(protein_seq), frameshift_mutations





def write_fasta_with_metadata_from_df(df, output_file, reference_length):
    """
    Writes translated protein sequences to a FASTA file with metadata in the header.

    Args:
        df (pd.DataFrame): DataFrame containing Filename, Protein_Sequence, Phenotype, and Frameshift_Mutation.
        output_file (str): Path to save the FASTA file.
        reference_length (int): Length of the reference protein sequence.
    """
    with open(output_file, "w") as fasta_file:
        for _, row in df.iterrows():
            filename = row["Filename"]
            sequence = row["Protein_Sequence"] 
            phenotype = row["Phenotype"]
            frameshift_flag = row["Frameshift_Mutation"]
            seq_len = row["seq_len"]
            

            # Construct FASTA header
            header = f">{filename} | {phenotype} | {seq_len} | Frameshift: {frameshift_flag}"
            fasta_file.write(header + "\n")
            fasta_file.write(sequence + "\n")



