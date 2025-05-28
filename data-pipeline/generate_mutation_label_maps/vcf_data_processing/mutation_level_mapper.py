import pandas as pd
import os
from config import INSERTION_SITES_IN_ALIGNED_FASTA_DIR, INSERTION_SITES_IN_ALIGNED_FASTA_SUFFIX, GENE_OPERON_MAPS, GENE_START_COORDS, DRUGS


class MapVCFtoWHOVariants():
    def __init__(self, snp_vcf_df, protein_vcf_df, who_variants_df):
        self.snp_vcf_df = snp_vcf_df
        self.protein_vcf_df = protein_vcf_df
        self.who_variants_df = who_variants_df
        self.columns_for_protein_vcf_df = ['POSITION', 'REFERENCE_ALLELE', 'ALTERED_ALLELE', 'GENE', 'MUTATION_EFFECT', 'PROTEIN_CHANGE', 'vcf_variant']
        self.columns_for_snp_vcf_df = ['POSITION', 'REFERENCE_ALLELE', 'ALTERED_ALLELE', 'GENE', 'MUTATION_EFFECT', 'NUCLEOTIDE(S)_CHANGE', 'vcf_variant']
        # self.vcf_variant_positions = self.get_vcf_variant_positions()
        # self.who_variant_positions = self.get_who_variant_positions()

    def get_columns_for_pruned_isolate_vcf_df(self, columns_to_keep):
        # pruned_columns = ['CHROMOSOME', 'POSITION', 'VARIANT_ID', 'REFERENCE_ALLELE', 'ALTERED_ALLELE']
        columns_to_keep = columns_to_keep
        return columns_to_keep
    
    def prune_isolate_vcf_df(self, vcf_df, columns_to_keep):
        pruned_vcf_df = vcf_df[self.get_columns_for_pruned_isolate_vcf_df(columns_to_keep)]
        return pruned_vcf_df
        
    def get_vcf_variant_positions(self, vcf_df):
        vcf_variant_positions = self.vcf_df['POSITION'].tolist()
        return vcf_variant_positions
    
    def get_who_variant_positions(self):
        who_variant_positions = self.who_variants_df['genome_index'].tolist()
        return who_variant_positions
    
    def get_unique_drug_names(self):
        drug_names = self.who_variants_df['drug'].unique()
        return drug_names
    
    def get_columns_for_mapped_df(self):
        columns = ['POSITION'] + self.get_unique_drug_names()
        return columns
    
    def merge_protein_vcf_to_who_variants(self):
        pruned_protein_vcf_df = self.prune_isolate_vcf_df(self.protein_vcf_df, self.columns_for_protein_vcf_df)
        merged_df = pd.merge(self.who_variants_df, pruned_protein_vcf_df, left_on=['genome_index', 'variant', 'ref_nt', 'alt_nt'], 
                      right_on=['POSITION', 'vcf_variant', 'REFERENCE_ALLELE', 'ALTERED_ALLELE'])
        
        return merged_df
    
    def merge_single_snp_vcf_to_who_variants(self):
        pruned_single_snp_vcf_df = self.prune_isolate_vcf_df(self.snp_vcf_df, self.columns_for_snp_vcf_df)
        merged_df = pd.merge(self.who_variants_df, pruned_single_snp_vcf_df, left_on=['genome_index', 'ref_nt', 'alt_nt'], 
                      right_on=['POSITION', 'REFERENCE_ALLELE', 'ALTERED_ALLELE'])
        
        return merged_df
    
    def postprocess_merged_df(self, merged_df):
        merged_df.rename(columns={'POSITION': 'mutation_position', 'MUTATION_EFFECT': 'mutation_effect'}, inplace=True)
        columns_to_keep = ['drug', 'variant', 'confidence', 'gene', 'mutation_position', 'mutation_effect']
        merged_df = self.prune_isolate_vcf_df(merged_df, columns_to_keep)
        merged_df.drop_duplicates(inplace=True)

        return merged_df
    
    def map_vcf_variants_to_who_drug_resistance(self, df):
        """
        Maps WHO variant-level drug resistance information into a matrix:
        - One row per variant (variant + gene + position + effect)
        - One column per drug for resistance status (R/S)
        - One column per drug for confidence string
        """
        merged_rows = {}

        # df should have drugs listed in DRUGS list
        df = df[df['drug'].isin(DRUGS)]    


        for _, row in df.iterrows():
            key = (
                str(row['variant']),
                str(row['gene']),
                str(row['mutation_position']),
                str(row['mutation_effect'])
            )

            drug = row['drug']
            confidence = row.get('confidence', '')

            if key in merged_rows:
                if '1) Assoc w R' in confidence or '2) Assoc w R - Interim' in confidence:
                    merged_rows[key][drug] = 'R'
                elif '3) Uncertain significance' in confidence:
                    merged_rows[key][drug] = 'N/A'
                elif '4) Not assoc w R' in confidence or '5) Not assoc w R - Interim' in confidence:
                    merged_rows[key][drug] = 'S'
                merged_rows[key][f"{drug}_confidence"] = confidence
            else:
                new_row = {
                    'variant': row['variant'],
                    'gene': row['gene'],
                    'mutation_position': row['mutation_position'],
                    'mutation_effect': row['mutation_effect'],
                }

                for d in DRUGS:
                    new_row[d] = ''
                    new_row[f"{d}_confidence"] = ''

                if '1) Assoc w R' in confidence or '2) Assoc w R - Interim' in confidence:
                    new_row[drug] = 'R'
                elif '3) Uncertain significance' in confidence:
                    new_row[drug] = 'N/A'
                elif '4) Not assoc w R - Interim' in confidence or '5) Not assoc w R' in confidence:
                    new_row[drug] = 'S'
                new_row[f"{drug}_confidence"] = confidence

                merged_rows[key] = new_row

        resistance_mapped_df = pd.DataFrame(list(merged_rows.values()))
        return resistance_mapped_df

    def create_gapped_mutation_position(self, df):
        # Ensure the ungapped_mutation_position column exists in df
        if 'mutation_position' not in df.columns:
            raise ValueError("Dataframe must contain a 'mutation_position' column")
        
        # Create the new empty column for gapped mutation positions
        df['gapped_mutation_position_pos_strand'] = None
        df['rel_gapped_mutation_position_pos_strand'] = None
        df['rel_gapped_mutation_position_neg_strand'] = None
        
        for idx, row in df.iterrows():
            gene = row.get('gene')
            if pd.isna(gene):
                raise ValueError("There's no gene column")
            
            # first map the gene to the operon and then use that gene
            gene = GENE_OPERON_MAPS.get(gene, gene)

            # Locate the corresponding gene folder in the sites directory
            gene_folder_path = os.path.join(INSERTION_SITES_IN_ALIGNED_FASTA_DIR, gene)
            if not os.path.exists(gene_folder_path) or not os.path.isdir(gene_folder_path):
                print("Gene folder does not exist. Skipping...")
                continue  # Skip if gene folder does not exist

            # print("\nGene folder: ", gene_folder_path)

            # Locate the insertion file inside the gene folder
            insertions_file_name = f"{gene}{INSERTION_SITES_IN_ALIGNED_FASTA_SUFFIX}.csv"
            insertion_file_path = os.path.join(gene_folder_path, insertions_file_name)
            if not os.path.exists(insertion_file_path):
                raise ValueError("There's no insertions file")
            
            # Read the insertion file
            insertion_df = pd.read_csv(insertion_file_path)
            
            # Ensure required columns exist in the insertion file
            required_cols = {'aln_idx', 'len_insertion'}
            if not required_cols.issubset(insertion_df.columns):
                print(f"Missing required columns in {insertions_file_name}. Skipping...")
                continue  # Skip if required columns are missing
            
            # Calculate the updated mutation position
            ungapped_mutation_pos = row['mutation_position']

            if ungapped_mutation_pos < GENE_START_COORDS[gene][0]:
                # different relative numbering for upstream variants
                df.at[idx, 'gapped_mutation_position_pos_strand'] = ungapped_mutation_pos
                df.at[idx, 'rel_gapped_mutation_position_pos_strand'] = ungapped_mutation_pos - GENE_START_COORDS[gene][0]

            else:
                # subtract gene start coord from the mutation position
                relative_mutation_pos_pos_strand = ungapped_mutation_pos - GENE_START_COORDS[gene][0]
                num_gaps = insertion_df[insertion_df['aln_idx'] <= relative_mutation_pos_pos_strand]['len_insertion'].sum()
                df.at[idx, 'gapped_mutation_position_pos_strand'] = ungapped_mutation_pos + num_gaps

                # adding 1 to make the relative position 1-indexed
                df.at[idx, 'rel_gapped_mutation_position_pos_strand'] = relative_mutation_pos_pos_strand + num_gaps + 1

            # add gaps to the end coord
            if GENE_START_COORDS[gene][-1] == "neg":
                if ungapped_mutation_pos > GENE_START_COORDS[gene][-2]:
                    # different relative numbering for upstream variants
                    df.at[idx, 'rel_gapped_mutation_position_neg_strand'] = GENE_START_COORDS[gene][-2] - ungapped_mutation_pos

                else:
                    total_gaps = insertion_df['len_insertion'].sum()
                    gapped_end_coord = GENE_START_COORDS[gene][-2] + total_gaps
                    gene_length = gapped_end_coord - GENE_START_COORDS[gene][0] + 1

                    # adding 1 to make the relative position 1-indexed
                    df.at[idx, 'rel_gapped_mutation_position_neg_strand'] = gene_length - df.at[idx, 'rel_gapped_mutation_position_pos_strand'] + 1
                    
        return df

    
    def create_isolate_vcf_to_who_mapped_df(self, is_protein_variant=True):
        '''
        Pivot the merged dataframe to have the drug names as the columns and the variant positions as the index
        '''
        merged_df = self.merge_protein_vcf_to_who_variants() if is_protein_variant else self.merge_single_snp_vcf_to_who_variants()
        
        if merged_df.empty:
            return merged_df, True

        merged_df = self.postprocess_merged_df(merged_df)
        mapped_df = self.map_vcf_variants_to_who_drug_resistance(merged_df)
        mapped_df = self.create_gapped_mutation_position(mapped_df)
        return mapped_df, False
