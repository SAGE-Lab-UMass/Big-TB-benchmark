class VCFColumns:
    """Defines VCF column names and data types."""
    
    def __init__(self):
        self.column_dtypes = {
            "#CHROM": str,
            "POS": int,
            "ID": str,
            "REF": str,
            "ALT": str,
            "QUAL": str,
            "FILTER": str,
            "INFO": str,
            "FORMAT": str,
            "SAMPLE": str,
            "ANN": str,
            "LOF": str,
            "NMD": str,
        }

    def get_column_dtypes(self):
        return self.column_dtypes

    def rename_columns(self):
        return {
            "#CHROM": "CHROMOSOME",
            "POS": "POSITION",
            "ID": "VARIANT_ID",
            "REF": "REFERENCE_ALLELE",
            "ALT": "ALTERED_ALLELE",
            "QUAL": "VARIANT_CALL_QUALITY_SCORE",
            "FILTER": "FILTER_STATUS",
            "INFO": "INFO_FIELDS",
            "FORMAT": "GENOME_DATA_FORMAT",
            "SAMPLE": "GENOME_SAMPLE",
        }
