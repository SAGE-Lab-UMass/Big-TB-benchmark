import os

# Directory containing multiple VCF files
# VCF_DIR = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs/annotated_isolates_vcf_csv_test"

VCF_DIR = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs/annotated_isolates_vcf_csv"

# Directory containing WHO variant data
WHO_VARIANTS_FILE = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/data/WHO_variants_data_2023/WHO_resistance_variants_all_2023.csv"

# Aligned insertion sites in fasta
INSERTION_SITES_IN_ALIGNED_FASTA_DIR = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_msa/results/aligned/sites"
INSERTION_SITES_IN_ALIGNED_FASTA_SUFFIX = "_group_1_combined_insertion_sites"
# "_group_1_normal_insertion_sites"

# Output directory for mapped mutation level CSV files
MUTATIONS_LEVEL_MAPS_DIR = os.path.join("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs", "vcf_who_mapped_csv_wo_s")
os.makedirs(MUTATIONS_LEVEL_MAPS_DIR, exist_ok=True)

# Output directory for compiled isolate level CSV files
ISOLATE_LEVEL_MAPPINGS_FILE_BIN = os.path.join("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs", "isolate_level_mapping_bin_w_na.csv")
ISOLATE_LEVEL_MAPPINGS_FILE = os.path.join("/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs", "isolate_level_mapping_w_na.csv")

GENE_OPERON_MAPS = {
    "rrs": "rrs-rrl",
    "rrl": "rrs-rrl",
    "embC": "embCAB",
    "embA": "embCAB",
    "embB": "embCAB",
    # "ethA": "ethAR",
    # "ethR": "ethAR",
    "fabG1": "fabG1-inhA",
    "inhA": "fabG1-inhA",
    "gyrB": "gyrBA",
    "gyrA": "gyrBA",
    "rpoB": "rpoBC",
    "rpoC": "rpoBC",
}

GENE_START_COORDS = {   
    "rrs-rrl": [1471846, 1476795, "pos"],  # 
    "Rv0678": [778990, 779487, "pos"],     # 
    "tlyA": [1917940, 1918746, "pos"],     
    "embCAB": [4239763, 4249910, "pos"],   # embCAB operon
    # "ethAR": [4326004, 4328199, "neg"],
    "ethA" : [4326004, 4327473, "neg"],
    "ethR": [4327549, 4328199, "pos"],    # regulatory gene, negatively regulates ethA
    "fabG1-inhA": [1673340, 1675111, "pos"], # upstream downstream promoter regions considered for inhA
    "katG": [2153889, 2156111, "neg"], # 
    "eis": [2714024, 2715432, "neg"],  # upstream downstream promoter regions considered for eis 
    "gyrBA": [5140, 9918, "pos"],      # gyrBA operon  
    "rplC": [800809, 801462, "pos"],   # 
    "pncA": [2288681, 2289241, "neg"], # 
    "rpoBC": [759707, 767420, "pos"],  # rpoBC operon 
    "gid": [4407528, 4408202, "neg"], # 
    "rpsL": [781560, 781934, "pos"], # 
}

DRUGS = [
    'Isoniazid',    # INH
    'Rifampicin',   # RIF
    'Ethambutol',   # EMB
    'Pyrazinamide', # PZA
    'Streptomycin', # (SM) less commonly used now as very toxic
    'Kanamycin', 
    'Amikacin', 
    'Capreomycin', 
    # 'OFLOXACIN',    # FQ
    'Levofloxacin', # FQ 
    'Moxifloxacin', # FQ 
    # 'CIPROFLOXACIN', # FQ
    'Ethionamide',
    # 'BEDAQUILINE',
    # 'CLOFAZIMINE',
    # 'DELAMANID',
    # 'LINEZOLID',
]