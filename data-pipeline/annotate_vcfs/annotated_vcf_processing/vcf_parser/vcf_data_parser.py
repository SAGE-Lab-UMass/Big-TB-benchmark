import pandas as pd
import io

from vcf_parser.vcf_columns_parser import VCFColumns

class VCFDataParser:
    """Class for reading VCF files and converting them into DataFrames."""

    def __init__(self):
        self.column_data_types = VCFColumns()

    def read_vcf_file(self, file_path):
        """Reads a VCF file and returns a DataFrame."""
        with open(file_path, "r") as f:
            lines = [line for line in f if not line.startswith("##")]

        vcf_buffer = io.StringIO("".join(lines))
        vcf_df = pd.read_csv(
            vcf_buffer,
            dtype=self.column_data_types.get_column_dtypes(),
            sep="\t"
        ).rename(columns=self.column_data_types.rename_columns())

        return vcf_df
