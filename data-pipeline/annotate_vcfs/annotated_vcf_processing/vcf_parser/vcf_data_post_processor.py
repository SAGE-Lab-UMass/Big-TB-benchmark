import pandas as pd
import re

import ipdb

class PostprocessVCFDf:
    """Class for efficient and automated post-processing of VCF data."""

    def __init__(self, vcf_data_df: pd.DataFrame):
        self.vcf_data_df = vcf_data_df
        self.info_fields = self._extract_info_keys()
        self.ann_fields = self._define_ann_keys()
        self.multivalued_info_fields = self._define_multivalued_info_keys()
        self._exceptional_ann_fields_rows = []

    def _extract_info_keys(self):
        """
        Extracts unique INFO field keys dynamically.
        """
        info_keys = set()
        non_empty_info = self.vcf_data_df["INFO_FIELDS"].dropna()

        info_keys.update(pair.split("=")[0] for entry in non_empty_info for pair in entry.split(";"))

        return {key: [] for key in sorted(info_keys)}

    def _define_ann_keys(self):
        """
        Returns the predefined ANN field structure.
        """
        return {
            "Allele": str, "Annotation": str, "Annotation_Impact": str,
            "Gene_Name": str, "Gene_ID": str, "Feature_Type": str,
            "Feature_ID": str, "Transcript_BioType": str, "Rank": str,
            "HGVS.c": str, "HGVS.p": str, "cDNA.pos / cDNA.length": str,
            "CDS.pos / CDS.length": str, "AA.pos / AA.length": str,
            "Distance": str, "ERRORS / WARNINGS / INFO": str
        }

    def _define_multivalued_info_keys(self):
        """
        Defines the multivalued INFO fields with correct data types.
        """
        return {
            "BC": int, "QP": int, "AC": int, "AF": float, "SVLEN": str
        }

    def _assign_unique_variant_id(self):
        """Assigns a unique variant ID for each entry."""
        self.vcf_data_df["VARIANT_ID"] = self.vcf_data_df["CHROMOSOME"] + "_" + self.vcf_data_df["POSITION"].astype(str)

    def _assign_na_to_missing_quality_score(self):
        """Replaces missing quality scores with NaN."""
        self.vcf_data_df["VARIANT_CALL_QUALITY_SCORE"] = self.vcf_data_df["VARIANT_CALL_QUALITY_SCORE"].replace(".", None)

    def _split_info_fields(self):
        """Efficiently extracts INFO fields into separate columns."""
        info_dict = {key: [] for key in self.info_fields}

        for entry in self.vcf_data_df["INFO_FIELDS"].fillna(""):
            entry_dict = dict(pair.split("=") if "=" in pair else (pair, True) for pair in entry.split(";"))
            for key in self.info_fields:
                info_dict[key].append(entry_dict.get(key, None))

        self.vcf_data_df = self.vcf_data_df.join(pd.DataFrame(info_dict)).drop(columns=["INFO_FIELDS"])

    def _extract_ann_column(self):
        """Efficiently extracts ANN column from INFO_FIELDS."""
        self.vcf_data_df["ANN"] = self.vcf_data_df["INFO_FIELDS"].str.extract(r"ANN=([^;]+)", expand=False)

    # def _split_ann_fields(self):
    #     """Splits ANN field into predefined columns efficiently and removes inconsistent rows."""
    #     ann_data = self.vcf_data_df["ANN"].fillna("").str.split("|", expand=True)

    #     # Identify rows with incorrect number of ANN fields
    #     incorrect_rows = ann_data.index[ann_data.apply(lambda x: len(x.dropna()) != len(self.ann_fields), axis=1)].tolist()
    #     self._exceptional_ann_fields_rows.extend(incorrect_rows)

    #     print("Shape before:", ann_data.shape[0])
    #     print("list:", self._exceptional_ann_fields_rows)
    #     # Drop rows with inconsistent ANN fields
    #     ann_data.drop(index=self._exceptional_ann_fields_rows, inplace=True)
    #     self.vcf_data_df.drop(index=self._exceptional_ann_fields_rows, inplace=True)

    #     print("Shape after:", ann_data.shape[0])
    #     print("Expected length:", len(self.ann_fields))

    #     incorrect_rows = ann_data.index[ann_data.apply(lambda x: len(x.dropna()) != len(self.ann_fields), axis=1)].tolist()
    #     self._exceptional_ann_fields_rows.extend(incorrect_rows)
    #     print("new list:", self._exceptional_ann_fields_rows)

    #     # Assign correct column names
    #     ann_data.columns = self.ann_fields.keys()
    #     self.vcf_data_df = self.vcf_data_df.join(ann_data).drop(columns=["ANN"])
        
    # Function to separate ANN fields into separate columns
    def _split_ann_fields(self):
        # Find indices of rows with empty or missing 'ANN'
        rows_to_drop = self.vcf_data_df[
            self.vcf_data_df['ANN'].isna() | (self.vcf_data_df['ANN'] == "")
        ].index

        # Log how many rows will be dropped
        print(f"\nDropping {len(rows_to_drop)} rows with empty or missing 'ANN' field.")

        # Drop the rows and reset index
        self.vcf_data_df.drop(index=rows_to_drop, inplace=True)
        self.vcf_data_df.reset_index(drop=True, inplace=True)

        # first split the ANN column by ','
        ann_comma_split = self.vcf_data_df['ANN'].str.split(',')
        multi_annotation_entries = {}

        # Split the 'ANN' column by '|'
        ann_split = self.vcf_data_df['ANN'].str.split('|')
        ann_fields = self.ann_fields


        for idx, split_list in ann_comma_split.items():
            high_impact_entry = None

            if len(split_list) == 1:
                continue  # Skip if there's only one entry
            for entry in split_list:
                split_entry = entry.split('|')
                
                # Ensure the entry has enough elements
                if len(split_entry) < 3:
                    continue  

                impact = split_entry[2]

                if impact == 'HIGH':
                    high_impact_entry = entry  # Store full entry if it's the desired variant type
                    break
                elif impact == 'MODERATE':
                    high_impact_entry = entry
                elif high_impact_entry is None:
                    high_impact_entry = entry

            if high_impact_entry:
                multi_annotation_entries[idx] = high_impact_entry  # Ensure only one selection per row idx


        # Done for cases when ANN has multiple entries
        for idx, split_list in ann_split.items():
            if idx in multi_annotation_entries:
                ann_split[idx] = multi_annotation_entries[idx].split('|')
                split_list = multi_annotation_entries[idx].split('|')
            
            if len(split_list) != len(ann_fields):
                print(f"Row {idx} contains elements different from required ann_fields.")
                self._exceptional_ann_fields_rows.append(idx)
                ipdb.set_trace()

        # expand the split 'ANN' column into separate columns
        ann_split = pd.DataFrame([x for x in ann_split], columns=ann_fields.keys()) 

        # print("length of dropped df:", len(self.vcf_data_df))
        # print("length of ann split df:", len(ann_split))

        assert len(self.vcf_data_df) == len(ann_split)
        
        # Concatenate the split columns to the original dataframe
        self.vcf_data_df = pd.concat([self.vcf_data_df, ann_split], axis=1) 

        if len(self._exceptional_ann_fields_rows):
            self.vcf_data_df.drop(index=self._exceptional_ann_fields_rows, inplace=True)

        # Convert columns to specified data types
        for col, dtype in ann_fields.items():
            ann_split[col] = ann_split[col].astype(dtype)
        
        # Drop the original 'ANN' column
        self.vcf_data_df.drop('ANN', axis=1, inplace=True)

        return True
    

    def _convert_multivalued_info(self):
        """
        Converts multivalued INFO fields into lists of correct data types.
        """
        for column, dtype in self.multivalued_info_fields.items():
            if column in self.vcf_data_df.columns and self.vcf_data_df[column].astype(str).str.contains(",").any():
                self.vcf_data_df[column] = self.vcf_data_df[column].apply(
                    lambda x: [dtype(float(i)) for i in x.split(",")] if pd.notna(x) else x
                )

    def _assign_mutation_type(self):
        """Assigns mutation type based on extracted SVTYPE values."""
        self.vcf_data_df["MUTATION_TYPE"] = "SNP"

        if "SVTYPE" in self.vcf_data_df.columns:
            self.vcf_data_df.loc[self.vcf_data_df["SVTYPE"].isin(["INS", "DEL"]), "MUTATION_TYPE"] = "INDEL"

        if {"IC", "DC"}.issubset(self.vcf_data_df.columns):
            self.vcf_data_df.loc[(self.vcf_data_df["IC"] != "0") | (self.vcf_data_df["DC"] != "0"), "MUTATION_TYPE"] = "INDEL"

    def postprocess(self):
        """Executes all post-processing steps efficiently."""
        self._assign_unique_variant_id()
        self._assign_na_to_missing_quality_score()
        self._split_info_fields()
        # self._extract_ann_column()
        is_success = self._split_ann_fields()
        if not is_success:
            return self.vcf_data_df, is_success
        self._convert_multivalued_info()
        self._assign_mutation_type()
        return self.vcf_data_df, is_success
