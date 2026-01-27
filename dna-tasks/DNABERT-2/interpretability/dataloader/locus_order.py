# locus_order = [
#         "gyrB",  
#         "gyrA", 
#         "rpsL", 
#         "rpoB",
#         "rplC",  
#         "inhA",     # upstream downstream promoter regions considered for inhA
#         "katG",
#         "pncA",
#         "eis",     # upstream downstream promoter regions considered for eis
#         "tlyA", 
#         "rrs", 
#         "embB",
#         "ethA",
#         "gid"
# ]

# locus_order = [
#     "gyrBA",
#     "rpoBC",
#     # "Rv0678",
#     "rpsL",
#     "rplC",
#     "FabG1-inhA",
#     "rrs-rrl",
#     "tlyA",
#     "katG",
#     "pncA",
#     "eis",
#     "embCAB",
#     "ethAR",
#     "gid"
# ]

locus_order = [
    "gyrBA",
    "rpoBC",
    # "Rv0678",
    "rpsL",
    # "rplC",
    "fabG1-inhA",
    "rrs-rrl",
    "tlyA",
    "katG",
    "pncA",
    # "eis",
    "embCAB",
    "ethA",
    "ethR",
    "gid"
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

DRUG_TO_LOCI = {
    'ISONIAZID': ['inhA', 'katG'], # ran
    'RIFAMPICIN': ['rpoB', 'rpoC'], # ran1
    'ETHAMBUTOL': ['embC', 'embA', 'embB'], # ran1
    'PYRAZINAMIDE': ['pncA'], # ran1
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'], # ran1
    'KANAMYCIN': ['rrs'], # ip
    'AMIKACIN': ['rrs', 'eis'], # F
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'], # ran1
    'LEVOFLOXACIN': ['gyrB', 'gyrA'], # ran1
    'MOXIFLOXACIN': ['gyrB', 'gyrA'], # ran1
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'], # ip
    # Add more drugs and their corresponding loci as needed
} 

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
    
# GENE_OPERONIC_PAIR_MAPS = {
#     "rrs": "rrs-rrl",
#     "rrl": "rrs-rrl",
#     "embC": "embCAB",
#     "embA": "embCAB",
#     "embB": "embCAB",
#     # "ethA": "ethAR",
#     # "ethR": "ethAR",
#     "fabG1": "fabG1-inhA",
#     "inhA": "fabG1-inhA",
#     "gyrB": "gyrBA",
#     "gyrA": "gyrBA",
#     "rpoB": "rpoBC",
#     "rpoC": "rpoBC",
# }

GENE_OPERONIC_PAIR_MAPS = {}