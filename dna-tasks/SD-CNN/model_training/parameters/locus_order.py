LOCUS_ORDER = [
    "gyrB", 
    "gyrA", 
    "rpoB", # done1 - dim - seq - mdim - ip
    "rpoC", # done1 - dim - seq - mdim
    "rpsL", # done1 - dim - seq - mdim
    "fabG1",
    "inhA", #done1 - dim - seq - mdim - ip
    "rrs", # done1 - dim - seq - mdim
    "rrl", # done1 - dim - seq - mdim
    "tlyA", # done1 - dim - seq - mdim
    "katG", #done1 - dim - seq - mdim - ip
    "pncA", #done1 - dim - seq - mdim
    "eis", # done1 - dim - seq - mdim
    "embC", #done1 - dim - seq - mdim
    "embA", # done1 - dim - seq - mdim
    "embB", #done1 - dim - seq - mdim
    "ethA", # done1 - dim - seq - mdim
    "ethR", # done1 - dim - seq - mdim
    "gid" # done1 - dim - seq - mdim
]

DRUGS = [
    'ISONIAZID',    # INH
    'RIFAMPICIN',   # RIF
    'ETHAMBUTOL',   # EMB
    'PYRAZINAMIDE', # PZA
    'STREPTOMYCIN', # (SM) less commonly used now as very toxic
    'KANAMYCIN', 
    'AMIKACIN', 
    'CAPREOMYCIN', 
    'OFLOXACIN',    # FQ
    'LEVOFLOXACIN', # FQ 
    'MOXIFLOXACIN', # FQ 
    'CIPROFLOXACIN', # FQ
    'ETHIONAMIDE',
]

BASE_TO_COLUMN = {
    'A': 0, 
    'C': 1, 
    'T': 2, 
    'G': 3, 
    '-': 4,
}
    
DRUG_TO_LOCI = {
    'ISONIAZID': ['inhA', 'katG'], # resource error
    'RIFAMPICIN': ['rpoB', 'rpoC'], # ran
    'ETHAMBUTOL': ['embC', 'embA', 'embB'], # ran
    'PYRAZINAMIDE': ['pncA'], # ran
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'], # ran
    'KANAMYCIN': ['rrs'], # ran
    'AMIKACIN': ['rrs', 'eis'], # ran
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'], # ran
    'LEVOFLOXACIN': ['gyrB', 'gyrA'], # only single class issue for AUC - fix it
    'MOXIFLOXACIN': ['gyrB', 'gyrA'], # ran
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'], # ran
    # Add more drugs and their corresponding loci as needed
} 