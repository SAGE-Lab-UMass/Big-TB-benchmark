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
    "gyrB",
    "gyrA"
    "rpoB",
    "rpoC"
    # "Rv0678",
    "rpsL",
    # "rplC",
    "fabG1",
    "inhA"
    "rrs",
    "rrl"
    "tlyA",
    "katG",
    "pncA",
    # "eis",
    "embC",
    "embA",
    "embB",
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
    