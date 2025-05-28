import os
from config import VCF_EXTENSION, ANNOTATED_VCF_EXTENSION, CSV_EXTENSION

class VCFDataWriter:
    """Class for writing VCF DataFrame to CSV files."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_to_csv(self, df, file_name):
        """Writes DataFrame to CSV."""
        output_file_path = os.path.join(self.output_dir, file_name.replace(ANNOTATED_VCF_EXTENSION, CSV_EXTENSION))
        df.to_csv(output_file_path, sep="\t", index=False)
