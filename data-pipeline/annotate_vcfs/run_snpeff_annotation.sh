#!/bin/bash

# Set input file containing VCF paths
input_vcf_combined_codons_file_paths="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_data/isolate_vcf_normal_paths.txt"

# Define output file to store modified VCF paths
modified_vcf_paths="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/vcf_data/modified_vcf_paths_normal.txt"

# Ensure the input file exists
if [[ ! -f $input_vcf_combined_codons_file_paths ]]; then
    echo "Error: Input file $input_vcf_combined_codons_file_paths not found!"
    exit 1
fi

# Process each line in the input file and modify paths
echo "Generating modified VCF paths..."
rm -f $modified_vcf_paths  # Remove old file if it exists

# Read each line properly, handling spaces and special characters
echo "Processing VCF file paths..."
while IFS= read -r vcf_file || [[ -n $vcf_file ]]; do
    # Trim any whitespace
    vcf_file=$(echo $vcf_file | tr -d '\r' | xargs)

    # Skip empty lines
    if [[ -z $vcf_file ]]; then
        continue
    fi

    # Check if the file exists
    if [[ -f $vcf_file ]]; then
        # Replace .vcf with _combinedCodons.vcf
        modified_file="${vcf_file%.vcf}_combinedCodons.vcf"
        echo $modified_file >> $modified_vcf_paths
    else
        echo "Warning: File $vcf_file not found, skipping..."
    fi
done < $input_vcf_combined_codons_file_paths

# Ensure there are modified files to process
if [[ ! -s $modified_vcf_paths ]]; then
    echo "Error: No valid VCF files found after modification!"
    exit 1
fi

# Run snpEff annotation
echo "Running snpEff annotation..."
snpEff eff Mycobacterium_tuberculosis_gca_000195955 \
    -noStats -no-downstream -no-upstream \
    -fileList $modified_vcf_paths

# Cleanup temporary file (optional)
rm -f $modified_vcf_paths

echo "Annotation completed!"
