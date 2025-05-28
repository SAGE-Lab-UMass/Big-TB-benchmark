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