"""Locus and drug ordering copied from the DNABERT2 embedding pipeline."""

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
    "gid",
]

DRUGS = [
    "ISONIAZID",
    "RIFAMPICIN",
    "ETHAMBUTOL",
    "PYRAZINAMIDE",
    "STREPTOMYCIN",
    "KANAMYCIN",
    "AMIKACIN",
    "CAPREOMYCIN",
    "LEVOFLOXACIN",
    "MOXIFLOXACIN",
    "ETHIONAMIDE",
]

DRUG_TO_LOCI = {
    "ISONIAZID": ["inhA", "katG"],
    "RIFAMPICIN": ["rpoB", "rpoC"],
    "ETHAMBUTOL": ["embC", "embA", "embB"],
    "PYRAZINAMIDE": ["pncA"],
    "STREPTOMYCIN": ["rpsL", "rrs", "gid"],
    "KANAMYCIN": ["rrs"],
    "AMIKACIN": ["rrs", "eis"],
    "CAPREOMYCIN": ["rrs", "rrl", "tlyA"],
    "LEVOFLOXACIN": ["gyrB", "gyrA"],
    "MOXIFLOXACIN": ["gyrB", "gyrA"],
    "ETHIONAMIDE": ["inhA", "ethA", "ethR"],
}
