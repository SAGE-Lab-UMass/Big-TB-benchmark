import os
import ipdb
import pandas as pd
from config import VCF_DIR, WHO_VARIANTS_FILE, MUTATIONS_LEVEL_MAPS_DIR
from vcf_data_processing.file_loader import load_vcf_files, load_who_variants, load_vcf_files_generator
from vcf_data_processing.vcf_processor import process_vcf
from vcf_data_processing.who_processor import process_who_data
from vcf_data_processing.mutation_level_mapper import MapVCFtoWHOVariants

def main():
    # Load WHO data
    who_variants_df = load_who_variants(WHO_VARIANTS_FILE)
    who_variants_df = process_who_data(who_variants_df)

    # Load and process each VCF file
    # vcf_dataframes = load_vcf_files(VCF_DIR)
    
    for filename, vcf_df in load_vcf_files_generator(VCF_DIR):
    # vcf_dataframes.items():
        print(f"Processing {filename}...")
        protein_vcf, snp_vcf = process_vcf(vcf_df)

        # Map VCF data to WHO variants
        mapper = MapVCFtoWHOVariants(snp_vcf, protein_vcf, who_variants_df)

        # Generate variant mappings
        mapped_protein_variants, is_protein_df_empty = mapper.create_isolate_vcf_to_who_mapped_df(is_protein_variant=True)
        mapped_snp_variants, is_snp_df_empty = mapper.create_isolate_vcf_to_who_mapped_df(is_protein_variant=False)

        # Combine the results
        if is_protein_df_empty and is_snp_df_empty: 
            print("No matching variants at all!")
            
            # Write the filename to a log file
            with open("no_matching_variants_files.txt", "a") as f:
                f.write(f"{filename}\n")

            continue

        if is_protein_df_empty:
            print("No matching protein variants")
            final_mapped_df = mapped_snp_variants
        elif is_snp_df_empty:
            print("No matching snp variants")
            final_mapped_df = mapped_protein_variants
        else:
            final_mapped_df = pd.concat([mapped_protein_variants, mapped_snp_variants], ignore_index=True)

        # check number of rows in final_mapped_df
        # print(f"Number of rows in final_mapped_df: {final_mapped_df.shape[0]}")


        # Save the mapped file
        output_file = os.path.join(MUTATIONS_LEVEL_MAPS_DIR, f"{filename}")
        output_file = output_file.replace("_combinedCodons", "")
    
        final_mapped_df.to_csv(output_file, index=False)
        print(f"Saved mapped data to {output_file}\n")
        

if __name__ == "__main__":
    main()
