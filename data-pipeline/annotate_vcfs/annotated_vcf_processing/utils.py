import os

def get_vcf_files(directory, extension=".vcf"):
    """Retrieve all VCF files from a directory."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]
