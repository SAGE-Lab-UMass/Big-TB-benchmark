import os
import pandas as pd

from config import ANNOTATED_VCF_DIR, OUTPUT_VCF_DIR, ANNOTATED_VCF_EXTENSION
from utils import get_vcf_files
from vcf_parser.vcf_data_parser import VCFDataParser
from vcf_parser.vcf_data_post_processor import PostprocessVCFDf
from vcf_parser.vcf_data_writer import VCFDataWriter

def process_vcf_files(input_dir, output_dir):
    """Processes all VCF files in a directory and writes results to CSV."""
    
    vcf_files = get_vcf_files(input_dir, ANNOTATED_VCF_EXTENSION)
    vcf_parser = VCFDataParser()
    vcf_writer = VCFDataWriter(output_dir)

    for vcf_file in vcf_files:
        print(f"Processing file: {vcf_file}")
        vcf_df = vcf_parser.read_vcf_file(vcf_file)

        # if vcf_df has only one row, skip it
        if len(vcf_df) <= 1:
            print(f"Skipping file {vcf_file} due to insufficient data.")
            continue

        processor = PostprocessVCFDf(vcf_df)
        processed_df, is_success = processor.postprocess()
        if not is_success:
            print(f"Skipping file {vcf_file} due to processing errors.")
            # write this file path to a separate file
            with open(os.path.join(output_dir, "failed_vcf_files.txt"), "a") as f:
                f.write(f"{vcf_file}\n")
            continue
        else:
            print(f"Successfully processed file: {vcf_file}")
            # write this file path to a separate file
            with open(os.path.join(output_dir, "success_vcf_files.txt"), "a") as f:
                f.write(f"{vcf_file}\n")

        vcf_writer.write_to_csv(processed_df, os.path.basename(vcf_file))

# Run processing for both directories
process_vcf_files(ANNOTATED_VCF_DIR, OUTPUT_VCF_DIR)
