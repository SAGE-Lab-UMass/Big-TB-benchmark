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

drugs = [
    'RIFAMPICIN', 
    'ISONIAZID', 
    'PYRAZINAMIDE',
    'ETHAMBUTOL', 
    'STREPTOMYCIN', 
    'LEVOFLOXACIN',
    'CAPREOMYCIN', 
    'AMIKACIN', 
    'MOXIFLOXACIN',
    # 'OFLOXACIN', 
    'KANAMYCIN', 
    'ETHIONAMIDE',
    # 'CIPROFLOXACIN'
]

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

BASE_TO_COLUMN = {
    'A': 0, 
    'C': 1, 
    'T': 2, 
    'G': 3, 
    '-': 4,
}
    