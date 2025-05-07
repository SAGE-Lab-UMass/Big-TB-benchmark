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
    