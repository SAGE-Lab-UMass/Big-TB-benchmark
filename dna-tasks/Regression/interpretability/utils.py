import os

WHO_DRUGS_AVAILABLE  = [
    'Isoniazid',    
    'Rifampicin',   
    'Ethambutol',   
    'Pyrazinamide', 
    'Streptomycin', 
    'Kanamycin', 
    'Amikacin', 
    'Capreomycin', 
    'Levofloxacin', 
    'Moxifloxacin', 
    'Ethionamide',
]

GENE_OPERONIC_PAIR_MAPS = {}

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