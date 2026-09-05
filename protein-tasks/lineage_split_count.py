
from pathlib import Path
import csv
from collections import Counter, defaultdict

base = Path('/project/pi_annagreen_umass_edu/mahbuba/Big-TB-benchmark/protein-tasks/data/latest/lineage_splits_all_train')
out_path = base / 'lineage_split_counts_summary.csv'
min_count = 50
rows = []

for drug_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
    drug = drug_dir.name
    files = sorted(drug_dir.glob('heldout_lineage_*.csv'))
    if not files:
        continue

    global_counts = None
    for f in files:
        held = f.stem.replace('heldout_lineage_', '')
        split_counts = defaultdict(Counter)
        lineage_counts = defaultdict(Counter)

        with f.open() as fh:
            for row in csv.DictReader(fh):
                split = row['split']
                lineage = str(row['Lineage'])
                pheno = row['Phenotype']
                split_counts[split][pheno] += 1
                lineage_counts[lineage][pheno] += 1

        if global_counts is None:
            global_counts = lineage_counts

        test_r = split_counts['test']['R']
        test_s = split_counts['test']['S']
        train_r = split_counts['train']['R']
        train_s = split_counts['train']['S']

        rows.append({
            'drug': drug,
            'row_type': 'heldout_split',
            'lineage': held,
            'n': test_r + test_s,
            'R': test_r,
            'S': test_s,
            'train_n': train_r + train_s,
            'train_R': train_r,
            'train_S': train_s,
            'eligible_min50_RS_train_and_test': test_r >= min_count and test_s >= min_count and train_r >= min_count and train_s >= min_count,
        })

    for lineage in sorted(global_counts, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x))):
        ph = global_counts[lineage]
        rows.append({
            'drug': drug,
            'row_type': 'all_rows_by_lineage',
            'lineage': lineage,
            'n': ph['R'] + ph['S'],
            'R': ph['R'],
            'S': ph['S'],
            'train_n': '',
            'train_R': '',
            'train_S': '',
            'eligible_min50_RS_train_and_test': '',
        })

fieldnames = ['drug','row_type','lineage','n','R','S','train_n','train_R','train_S','eligible_min50_RS_train_and_test']
with out_path.open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(out_path)
print('Eligible held-out splits:')
for r in rows:
    if r['row_type'] == 'heldout_split' and r['eligible_min50_RS_train_and_test']:
        print(f"{r['drug']}: L{r['lineage']} test n={r['n']} R={r['R']} S={r['S']} | train n={r['train_n']} R={r['train_R']} S={r['train_S']}")
