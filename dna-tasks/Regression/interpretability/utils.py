import os

WHO_DRUGS_AVAILABLE  = [
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

DRUGS_UNAVAILABLE = ["Ciprofloxacin", "Ofloxacin"]

GENE_OPERONIC_PAIR_MAPS = {
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

def create_output_dir(output_dir, drug):
    """
    Create output directory if it does not exist

    Parameters
    ----------
    output_dir: str
        Path to output directory

    drug: str
        Name of the drug to create a subdirectory for

    Returns
    -------
    str
        Path to the created output directory
    """
    # Create the base output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the drug-specific subdirectory
    # if drug not upper case, convert it to uppercase letters
    output_path = os.path.join(output_dir, drug.upper())
    saved_models_path = os.path.join(output_path, "saved_models")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(saved_models_path, exist_ok=True)
    
    return output_path, saved_models_path