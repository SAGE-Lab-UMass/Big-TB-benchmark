#!/bin/bash

# Assign the gene and group
group="group_1"
gene="rrs-rrl"  # Example: you can change this value
input_vcf_paths="path_list/isolate_vcf_normal_paths.txt"
secondary_vcf_paths="path_list/isolate_vcf_cryptic_paths.txt"
genbank_file_path="genbank_reference_data/genome.gbff"
msa_type="unaligned"
file_type="combined"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gene) gene="$2"; shift ;;  # Override gene variable
        -gr|--group) group="$2"; shift ;;  # Optional: Allow overriding group as well
        -f|--file_type) file_type="$2"; shift ;;  # Optional: Allow overriding file_type as well
        #-s|--sense) sense="$2"; shift ;;  # Optional: Allow overriding sense as well
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo $input_vcf_paths

# Use Python to import the gene_coords dictionary and extract start and end coordinates
start_end_sense=$(python -c "
from locus_coords.gene_coords import gene_coords

group = '${group}'
gene = '${gene}'

# Extract start and end coordinates
coords = gene_coords.get(group, {}).get(gene, [])
if coords:
    print(f'{coords[0]} {coords[-2]} {coords[-1]}')
else:
    exit(1)
")

# Check if start_end is empty (gene not found)
if [ -z "$start_end_sense" ]; then
    echo "Error: Start and End coordinates not found for gene $gene in $group"
    exit 1
fi

# Split the start_end into start and end variables
start=$(echo $start_end_sense | cut -d' ' -f1)
end=$(echo $start_end_sense | cut -d' ' -f2)
sense=$(echo $start_end_sense | cut -d' ' -f3)

# Output file name
output_file="${gene}_${group}_${file_type}.fasta"

# Python command to run the MSA script
python scripts/make_unaligned_msa.py -f $input_vcf_paths -start $start -end $end -sense $sense -o results/$msa_type/$output_file -g $genbank_file_path --resistance-groups $group --gene $gene --f2 $secondary_vcf_paths