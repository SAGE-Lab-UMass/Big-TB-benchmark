import os
import pandas as pd
import ipdb


DRUGS_LIST = [
    'Amikacin', 'Bedaquiline', 'Capreomycin', 'Clofazimine', 'Delamanid', 
    'Ethambutol', 'Ethionamide', 'Isoniazid', 'Kanamycin', 'Levofloxacin', 
    'Linezolid', 'Moxifloxacin', 'Pyrazinamide', 'Rifampicin', 'Streptomycin'
]

def get_drug_columns(df):
    """
    Extract drug columns from the dataframe, assuming the first few columns are metadata.
    Adjust the starting index as necessary based on the dataset structure.
    """
    return [drug for drug in DRUGS_LIST if drug in df.columns]

def process_isolate(file_path):
    """
    Process a single isolate CSV file and determine resistance profile.
    """
    print(f"Processing file: {file_path}")

    # if df = pd.read_csv(file_path) throws error, then return None
    try:
        df = pd.read_csv(file_path, na_values=[''], keep_default_na=False)
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty. Skipping.")
        # write this file path to a file called empty_files.txt
        with open("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/vcf_who_mapping/outputs/failed_maps.txt", "a") as f:
            f.write(file_path + "\n")
        return None
    
    # Extract isolate name from file name
    isolate_name = os.path.basename(file_path).replace(".csv", "")
    # if isolate_name == "IDR1100020842":
    #     print(f"isolate name is {isolate_name}")

    # Get drug columns
    drug_columns = get_drug_columns(df)

    # Initialize resistance profile
    resistance_profile = {drug: " " for drug in drug_columns}

    # Check for resistance (R/S) in each drug column
    for drug in drug_columns:
        if "R" in df[drug].values:
            resistance_profile[drug] = "R"
        elif "S" in df[drug].values:
            resistance_profile[drug] = "S"
        elif "N/A" in df[drug].values:
            resistance_profile[drug] = "N/A"

    # resistance_profile = {drug: " " for drug in drug_columns}

    # # Check for resistance (R/S) in each drug column
    # for drug in drug_columns:
    #     if drug == "Levofloxacin" and isolate_name == "IDR1100020842":
    #         print(f"Drug values: {df[drug].values}")

    #         # positions in the drug values having 'S'
    #         s_positions = [i+2 for i, val in enumerate(df[drug].values) if val == "S"]
    #         print(f"Positions with 'S': {s_positions}")
    #         ipdb.set_trace()
    #     if "R" in df[drug].values:
    #         resistance_profile[drug] = "R"
    #     elif "N/A" in df[drug].values:
    #         print(f"Skipping {drug} for isolate {isolate_name} (N/A found)")
    #         ipdb.set_trace()
    #         continue
    #     elif "S" in df[drug].values:
    #         resistance_profile[drug] = "S"

    # Return processed data for this isolate
    return {"Isolate": isolate_name, **resistance_profile}

def convert_to_binary(df):
    """
    Convert R/S notation to binary (R -> 1, S -> 0).
    """
    return df.replace({"R": 1, "S": 0})

def compile_resistance_profiles(directory, binary_format_output=False):
    """
    Process all CSV files in a directory and compile the resistance profile into a single CSV.
    """
    # Get all CSV files in the directory
    csv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]

    # Process each isolate file
    # isolate_data = [result for file in csv_files if (result := process_isolate(file)) is not None]
    isolate_data = list(filter(None, (process_isolate(file) for file in csv_files)))

    # Convert list to DataFrame
    final_df = pd.DataFrame(isolate_data)

    # Convert to binary if required
    if binary_format_output:
        final_df = convert_to_binary(final_df)

    return final_df

def write_to_csv(df, output_file):
    """
    Write the DataFrame to a CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Isolate level mappings saved to {output_file}")

