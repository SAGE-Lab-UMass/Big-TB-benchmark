# locus_order = [
#     "gyrBA",
#     "rpoBC",
#     # "Rv0678",
#     "rpsL",
#     # "rplC",
#     "fabG1-inhA",
#     "rrs-rrl",
#     "tlyA",
#     "katG",
#     "pncA",
#     # "eis",
#     "embCAB",
#     "ethA",
#     "ethR",
#     "gid"
# ]

locus_order = [
    "gyrB", # done1 - dim - seq - mdim
    "gyrA", # done1 - dim - seq - mdim
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
    'ISONIAZID',
    'RIFAMPICIN', 
    'ETHAMBUTOL',
    'PYRAZINAMIDE',
    'STREPTOMYCIN', 
    'KANAMYCIN',
    'AMIKACIN', 
    'CAPREOMYCIN', 
    'LEVOFLOXACIN',
    'MOXIFLOXACIN',
    'ETHIONAMIDE',
]

# DRUGS = [
#     'Isoniazid',    # INH
#     'Rifampicin',   # RIF
#     'Ethambutol',   # EMB
#     'Pyrazinamide', # PZA
#     'Streptomycin', # (SM) less commonly used now as very toxic
#     'Kanamycin', 
#     'Amikacin', 
#     'Capreomycin', 
#     # 'OFLOXACIN',    # FQ
#     'Levofloxacin', # FQ 
#     'Moxifloxacin', # FQ 
#     # 'CIPROFLOXACIN', # FQ
#     'Ethionamide',
#     # 'BEDAQUILINE',
#     # 'CLOFAZIMINE',
#     # 'DELAMANID',
#     # 'LINEZOLID',
# ]

BASE_TO_COLUMN = {
    'A': 0, 
    'C': 1, 
    'T': 2, 
    'G': 3, 
    '-': 4,
}
    
DRUG_TO_LOCI = {
    'ISONIAZID': ['inhA', 'katG'], # complete, mean_d ran, mean_s ran, pca train complete, mean_seq_dim train ran
    'RIFAMPICIN': ['rpoB', 'rpoC'], # complete, mean_d ran, mean_s ran, pca train complete, mean_seq_dim train ran
    'ETHAMBUTOL': ['embC', 'embA', 'embB'], # ran, 
    'PYRAZINAMIDE': ['pncA'], # complete, pca train complete, mean_d ran, mean_s ran, 
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'], # complete, pca train complete, mean_d ran, mean_s ran, mean_seq_dim train ran
    'KANAMYCIN': ['rrs'], # complete, pca train ran
    'AMIKACIN': ['rrs', 'eis'], # complete, mean_d ran, mean_s ran, pca train complete
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'], # ran
    'LEVOFLOXACIN': ['gyrB', 'gyrA'], # complete, pca train complete, mean_d ran, mean_s ran
    'MOXIFLOXACIN': ['gyrB', 'gyrA'], # complete, pca train complete
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'], # complete, pca train complete, mean_d ran, mean_s ran
} 
