gene_coords = {   # [] denotes the start and end coordinates of the gene
    "group_1": {
        "rrs-rrl": [1471846, 1476795, "pos"],  # 
        "Rv0678": [778990, 779487, "pos"],     # 
        "tlyA": [1917940, 1918746, "pos"],     
        "embCAB": [4239763, 4249910, "pos"],   # embCAB operon
        "ethA" : [4326004, 4327473, "neg"],
        "ethR": [4327549, 4328199, "pos"],    # regulatory gene, negatively regulates ethA
        "fabG1-inhA": [1673340, 1675111, "pos"], # upstream downstream promoter regions considered for inhA
        "katG": [2153889, 2156111, "neg"], # 
        "eis": [2714024, 2715432, "neg"],  # upstream downstream promoter regions considered for eis 
        "gyrBA": [5140, 9918, "pos"],      # gyrBA operon  
        "rplC": [800809, 801462, "pos"],   # 
        "pncA": [2288681, 2289241, "neg"], # 
        "rpoBC": [759707, 767420, "pos"],  # rpoBC operon 
        "gid": [4407528, 4408202, "neg"], # 
        "rpsL": [781560, 781934, "pos"], # 
        # "inhA": [1674202, 1675111, "pos"], # added +100bp downstream of the gene to include the operon structure
        # "rpoB": [759707, 763325, "pos"],  # added +100bp upstream of the gene to include the operon structure
    },
    "group_WHO_MTB_mix": {
        "rrs-rrl": [1471846, 1476795], 
        "Rv0678": [778990, 779487], 
        "tlyA": [1917940, 1918746],
        "embCAB": [4239763, 4249910],   # embCAB operon
        "ethAR": [4326004, 4328199],    # combined ethA and ethR (not on the same operon)
        "fabG1-inhA": [1673340, 1675111], # upstream downstream promoter regions considered for inhA
        "katG": [2153889, 2156111],
        "eis": [2714024, 2715432],  # upstream downstream promoter regions considered for eis
        "gyrBA": [5140, 9918],      # gyrBA operon
        "rplC": [800809, 801462],   
        "pncA": [2288681, 2289241], 
        "rpoBC": [759707, 767420],  # rpoBC operon
        "gid": [4407528, 4408202], 
        "rpsL": [781560, 781934], 
        "acpM-kasA": [2517671, 2519465], # acpM-kasA operon
        "gid": [4407528, 4408202],
        "rpsA": [1833542, 1834987],     
        "clpC1": [4038158, 4040704],        
        "aftB-ubiA": [4266953, 4269833],   # combined aftB and ubiA (not on the same operon)
        "oxyR-ahpC": [2725471, 2726880],   # oxyR-ahpC operon
        "panD": [4043862, 4044281],        # check with panC
    },
    "group_1_2": {  # incomplete
        "rrs-rrl": [1471846, 1476795], 
        "Rv0678": [778990, 779487], 
        "tlyA": [1917940, 1918746],
        "embCAB": [4239763, 4249910],   # embCAB operon
        "ethAR": [4326004, 4328199],    # combined ethA and ethR (not on the same operon)
        "FabG1-inhA": [1673340, 1675111], # upstream downstream promoter regions considered for inhA
        "katG": [2153889, 2156111],
        "eis": [2714024, 2715432],  # upstream downstream promoter regions considered for eis
        "gyrBA": [5140, 9918],      # gyrBA operon
        "rplC": [800809, 801462],   
        "pncA": [2288681, 2289241], 
        "rpoBC": [759707, 767420],  # rpoBC operon
        "gid": [4407528, 4408202], 
        "rpsL": [781560, 781934], 
        "bacA": [2062809, 2064728],        
        "ccsA": [619891, 620865],
        "Rv2477c": [2782366, 2784042],
        "whiB6": [4338171, 4338521],
        "whiB7": [3568401, 3568679],
        "atpBE": [1460144, 1461390],    # atpBE operon
    },
    "group_2": {
        "lpqB": [0, 0],        # Add specific coordinates
        "mmpL5": [0, 0],       # Add specific coordinates
        "mmpS5": [0, 0],       # Add specific coordinates
        "mtrA": [0, 0],        # Add specific coordinates
        "mtrB": [0, 0],        # Add specific coordinates
        "pepQ": [0, 0],        # Add specific coordinates
        "Rv0678": [778990, 779487],
        "Rv1979c": [0, 0],     # Add specific coordinates
        "rrl": [1476796, 1479848],
        "Rv2680": [0, 0],      # Add specific coordinates
        "Rv2681": [0, 0],      # Add specific coordinates
        "tlyA": [1917940, 1918746],
        "abG1biA": [0, 0],     # Add specific coordinates
        "abiB": [0, 0],        # Add specific coordinates
        "fbiC": [0, 0],        # Add specific coordinates
        "fgd1": [0, 0],        # Add specific coordinates
        "Rv2983": [0, 0],      # Add specific coordinates
        "ddn": [0, 0],         # Add specific coordinates
        "ndh": [0, 0],         # Add specific coordinates
        "aftB": [0, 0],        # Add specific coordinates
        "embA": [0, 0],        # Add specific coordinates
        "embB": [4246514, 4246615],
        "embC": [0, 0],        # Add specific coordinates
        "embR": [0, 0],        # Add specific coordinates
        "glpK": [0, 0],        # Add specific coordinates
        "Rv2752c": [0, 0],     # Add specific coordinates
        "ubiA": [0, 0],        # Add specific coordinates
        "ethA": [4326004, 4327473],
        "ethR": [0, 0],        # Add specific coordinates
        "inhA": [1673440, 1675111],
        "mshA": [0, 0],        # Add specific coordinates
        "Rv0565c": [0, 0],     # Add specific coordinates
        "Rv3083": [0, 0],      # Add specific coordinates
        "ahpC": [0, 0],        # Add specific coordinates
        "dnaA": [0, 0],        # Add specific coordinates
        "hadA": [0, 0],        # Add specific coordinates
        "katG": [2153889, 2156111],
        "Rv0010c": [0, 0],     # Add specific coordinates
        "Rv1129c": [0, 0],     # Add specific coordinates
        "Rv1258c": [0, 0],     # Add specific coordinates
        "gyrA": [737304, 739123],
        "gyrB": [752788, 755297],
        "rplC": [781840, 782352],
        "tsnR": [0, 0],        # Add specific coordinates
        "clpC1": [0, 0],       # Add specific coordinates
        "panD": [0, 0],        # Add specific coordinates
        "pncA": [2288091, 2288783],
        "PPE35": [0, 0],       # Add specific coordinates
        "rpsA": [0, 0],        # Add specific coordinates
        "Rv3236c": [0, 0],     # Add specific coordinates
        "sigE": [0, 0],        # Add specific coordinates
        "nusG": [0, 0],        # Add specific coordinates
        "rpoA": [0, 0],        # Add specific coordinates
        "rpoB": [759849, 763325],
        "rpoC": [0, 0],        # Add specific coordinates
        "gid": [1472657, 1473577],
        "rpsL": [781240, 781617]
    }
}

