locus_order = [
        "gyrB",  
        "gyrA", 
        "rpsL", 
        "rpoB",
        "rpoC",  
        "inhA",     # upstream downstream promoter regions considered for inhA
        "katG",
        "pncA",
        "eis",     # upstream downstream promoter regions considered for eis
        "tlyA", 
        "rrs", 
        "embB",
        "embA",
        "embC",
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


# Mapping drug names to indices for label extraction from phenotype data
DRUG_INDEX = {
    'ISONIAZID': 0,
    'RIFAMPICIN': 1,
    'ETHAMBUTOL': 2,
    'PYRAZINAMIDE': 3,
    'STREPTOMYCIN': 4,
    'KANAMYCIN': 5,
    'AMIKACIN': 6,
    'CAPREOMYCIN': 7,
    'LEVOFLOXACIN': 8,
    'MOXIFLOXACIN': 9,
    'ETHIONAMIDE': 10
}

BASE_TO_COLUMN = {
    'A': 0, 
    'C': 1, 
    'T': 2, 
    'G': 3, 
    '-': 4,
}

GENE_OPERONIC_PAIR_MAPS = {}