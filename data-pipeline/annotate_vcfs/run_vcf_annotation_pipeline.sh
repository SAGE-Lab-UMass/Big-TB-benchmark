#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Define the paths to the scripts
script1="./run_combine_codons.sh"
script2="./run_snpeff_annotation.sh"
# script3="./run_filter_by_R_confidence.sh"

# Ensure both scripts are executable
chmod +x $script1 $script2

# Run the first script
echo "Running first script: $script1"
bash $script1
echo "Codons combined successfully!\n"

# Run the second script after the first completes successfully
echo "Running second script: $script2"
bash $script2
echo "Annotation completed successfully!\n"

# Run the third script after the second completes successfully
# echo "Running third script: $script3"
# bash $script3
# echo "Annotated mutation positions for required genes selected successfully!\n"

echo "Pipeline execution completed successfully!"
