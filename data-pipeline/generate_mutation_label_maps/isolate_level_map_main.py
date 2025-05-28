from vcf_who_mapped_data_processing.isolate_level_mapper import compile_resistance_profiles, write_to_csv
from config import MUTATIONS_LEVEL_MAPS_DIR, ISOLATE_LEVEL_MAPPINGS_FILE, ISOLATE_LEVEL_MAPPINGS_FILE_BIN

def main():
    isolate_level_maps_df = compile_resistance_profiles(MUTATIONS_LEVEL_MAPS_DIR, binary_format_output=False)

    write_to_csv(isolate_level_maps_df, ISOLATE_LEVEL_MAPPINGS_FILE)

    isolate_level_maps_df_bin = compile_resistance_profiles(MUTATIONS_LEVEL_MAPS_DIR, binary_format_output=True)
    write_to_csv(isolate_level_maps_df_bin, ISOLATE_LEVEL_MAPPINGS_FILE_BIN)

if __name__ == "__main__":
    main()