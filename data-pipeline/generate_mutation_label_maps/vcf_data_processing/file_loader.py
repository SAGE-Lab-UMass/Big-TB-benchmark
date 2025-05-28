import os
import pandas as pd

def load_vcf_files(directory):
    """Loads all CSV VCF files from a directory."""
    vcf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    vcf_dataframes = {os.path.basename(file): pd.read_csv(file, sep='\t') for file in vcf_files}
    return vcf_dataframes

def load_who_variants(file_path):
    """Loads the WHO variants CSV file."""
    return pd.read_csv(file_path)

def load_vcf_files_generator(directory):
    """Generator function to load VCF files one at a time."""
    vcf_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    for filename in vcf_files:
        file_path = os.path.join(directory, filename)
        yield filename, pd.read_csv(file_path, sep='\t')  # Yield one file at a time

