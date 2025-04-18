import pandas as pd
import re

# === STEP 1: Load only the required colaumns from the large Excel file ===
file_path = "./data/WHO-UCN-TB-2023.7-eng.xlsx"
# Only select necessary columns to speed up loading
usecols = [
    "drug", "gene", "variant", "effect", "Present_R", "Present_S", "FINAL CONFIDENCE GRADING"
]
# Skip the first two rows, use row 3 (index 2) as header
df = pd.read_excel(
    file_path,
    sheet_name="Catalogue_master_file",
    header=2,  # 3rd row contains actual column names
    usecols=[
        "drug", "gene", "variant", "effect", "Present_R", "Present_S", "FINAL CONFIDENCE GRADING"
    ]
)
# Clean column names
df.columns = df.columns.str.strip()
print(df.columns)
# === STEP 2: Filter for valid protein substitutions ===
pattern = re.compile(r'(?P<gene>[A-Za-z0-9]+)_p\.(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2})')
df = df[df["variant"].astype(str).str.match(pattern)].copy()

# Filter for relevant genes
target_genes = {
    "katG", "eis", "gid", "pncA", "tlyA", "rpoB", "rpsL", "fabG1", "inhA",
    "embA", "embB", "embC", "gyrB", "gyrA", "ethA", "ethR", "Rv0678", "rrs-rrl", "rplC"
}
df["gene_from_variant"] = df["variant"].str.extract(r'^([A-Za-z0-9]+)_p\.')[0].str.lower()
df = df[df["gene_from_variant"].isin(g.lower() for g in target_genes)]

# === STEP 3: Extract mutation details ===
df[["aa_ref_3", "aa_pos", "aa_alt_3"]] = df["variant"].str.extract(
    r'_p\.(?P<aa_ref_3>[A-Z][a-z]{2})(?P<aa_pos>\d+)(?P<aa_alt_3>[A-Z][a-z]{2})'
)

# Translate to one-letter code
aa3_to_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E',
    'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V', 'Sec': 'U', 'Pyl': 'O', 'Asx': 'B', 'Glx': 'Z',
    'Xaa': 'X', 'Ter': '*'
}
df["aa_ref"] = df["aa_ref_3"].map(aa3_to_1)
df["aa_change"] = df["aa_alt_3"].map(aa3_to_1)
df["aa_pos"] = df["aa_pos"].astype(int)
df["one_letter_mutation"] = "p." + df["aa_ref"] + df["aa_pos"].astype(str) + df["aa_change"]

# === STEP 4: Finalize and export ===
final_cols = [
    "drug", "gene", "variant", "effect", "Present_R", "Present_S", "FINAL CONFIDENCE GRADING",
    "one_letter_mutation", "aa_pos", "aa_change", "aa_ref"
]
final_df = df[final_cols].rename(columns={
    "FINAL CONFIDENCE GRADING": "confidence"
})
final_df.to_csv("filtered_variants_output.csv", index=False)
print("Output saved to 'filtered_variants_output.csv'")