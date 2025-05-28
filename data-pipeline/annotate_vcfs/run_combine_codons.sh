#!/bin/bash

# Assign default values

input_vcf="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_data/text13.txt"
#input_vcf="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_data/isolate_vcf_test/IDR1100019254.vcf"
is_text_file=true  # Default: assume text file with list of VCF files

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input) input_vcf="$2"; shift ;;  # Input VCF file or text file with VCF paths
        -t|--is_text_file) is_text_file=true ;;  # Indicates input is a text file
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure input VCF file is provided
if [[ -z $input_vcf ]]; then
    echo "Error: Input VCF file or text file with VCF paths is required."
    echo "Usage: $0 -i <input_vcf_or_text_file_with_list> [-t]"
    exit 1
fi


# Determine output file name (only for single VCF files)
if [[ $is_text_file == false ]]; then
    output_file="${input_vcf%.vcf}_combinedCodons.vcf"
    echo "Processing $input_vcf -> $output_file"
    python variant_annotation/combine_codon_variants.py -i $input_vcf 
else
    echo "Processing multiple VCF files from list: $input_vcf"
    python variant_annotation/combine_codon_variants.py -i $input_vcf -t 
fi


