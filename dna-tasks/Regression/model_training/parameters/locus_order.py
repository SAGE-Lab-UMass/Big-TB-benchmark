locus_order = [
    "gyrB", 
    "gyrA", 
    "rpoB", 
    "rpoC", 
    "rpsL", 
    "fabG1",
    "inhA", 
    "rrs", 
    "rrl", 
    "tlyA", 
    "katG", 
    "pncA", 
    "eis", 
    "embC", 
    "embA", 
    "embB", 
    "ethA", 
    "ethR", 
    "gid" 
]

drugs = [
    'ISONIAZID',    
    'RIFAMPICIN',   
    'ETHAMBUTOL',   
    'PYRAZINAMIDE', 
    'STREPTOMYCIN', 
    'KANAMYCIN', 
    'AMIKACIN', 
    'CAPREOMYCIN', 
    'OFLOXACIN',   
    'LEVOFLOXACIN', 
    'MOXIFLOXACIN',  
    'CIPROFLOXACIN', 
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
    'ISONIAZID': ['inhA', 'katG'], 
    'RIFAMPICIN': ['rpoB', 'rpoC'], 
    'ETHAMBUTOL': ['embC', 'embA', 'embB'], 
    'PYRAZINAMIDE': ['pncA'], 
    'STREPTOMYCIN': ['rpsL', 'rrs', 'gid'],
    'KANAMYCIN': ['rrs'], 
    'AMIKACIN': ['rrs', 'eis'], 
    'CAPREOMYCIN': ['rrs', 'rrl', 'tlyA'], 
    'LEVOFLOXACIN': ['gyrB', 'gyrA'], 
    'MOXIFLOXACIN': ['gyrB', 'gyrA'], 
    'ETHIONAMIDE': ['inhA', 'ethA', 'ethR'], 
    # Add more drugs and their corresponding loci as needed
} 
    
GENE_OPERONIC_PAIR_MAPS = {}

GENE_START_COORDS = {   
    "tlyA": [1917840, 1918846, "pos"],     
    "ethA" : [4325904, 4327473, "neg"], # ethAR are together, just +100bp upstream
    "ethR": [4327549, 4328299, "pos"],    # just +100bp downstream, regulatory gene, negatively regulates ethA
    "katG": [2153789, 2156211, "neg"], # 
    "eis": [2714024, 2715432, "neg"],  # upstream downstream promoter regions considered for eis 
    "pncA": [2288581, 2289341, "neg"], # 
    "gid": [4407428, 4408302, "neg"], # 
    "rpsL": [781460, 782034, "pos"], # 
    "inhA": [1674202, 1675111, "pos"], # added +100bp downstream of the gene to include the operon structure
    "rpoB": [759707, 763325, "pos"],  # added +100bp upstream of the gene to include the operon structure
    "rpoC": [763370, 767420, "pos"],  # added +100bp downstream of the gene to include the operon structure 
    "gyrB": [5140, 7267, "pos"],  # added +100bp upstream of the gene to include the operon structure
    "gyrA": [7302, 9918, "pos"], # added +100bp downstream of the gene to include the operon structure 
    "embC": [4239763, 4243147, "pos"], # added +100bp upstream of the gene to include the operon structure
    "embA": [4243233, 4246517, "pos"], # no change
    "embB": [4246514, 4249910, "pos"], # added +100bp downstream of the gene to include the operon structure 
    "rrs": [1471746, 1473382, "pos"], # added +100bp upstream of the gene to include the operon structure 
    "rrl": [1473658, 1476895, "pos"], # added +100bp downstream of the gene to include the operon structure 
    "fabG1": [1673340, 1674183, "pos"], # added +100bp upstream of the gene to include the operon structure 
}