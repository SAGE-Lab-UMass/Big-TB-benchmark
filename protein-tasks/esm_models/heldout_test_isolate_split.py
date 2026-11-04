"""

Usage:  python esm_models/heldout_test_isolate_split.py --drug ethambutol --mode full
"""
import argparse, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--drug", required=True)
ap.add_argument("--mode", choices=["full", "pca"], required=True)
ap.add_argument("--out-root", default="data/latest/results/isolate_split")
args = ap.parse_args()

nb = json.load(open(HERE / "per_token_CNN_final.ipynb"))
ns = {"__name__": "esm_nb"}
for i in (1, 3, 4, 7):
    exec("".join(nb["cells"][i]["source"]), ns)
cell12 = "".join(nb["cells"][12]["source"])
exec("\n".join(re.sub(r"^# ?", "", l) for l in cell12.split("\n")), ns)

import pandas as pd
from data_utils import build_train_test_split

drug, mode = args.drug, args.mode
in_dim = 320 if mode == "full" else 10
gene = ns["single_drugs"][drug][0] if drug in ns["single_drugs"] else None

(train_files, y_train), (test_files, y_test) = build_train_test_split(drug, unique_isolates=True)
assert not set(train_files) & set(test_files), "isolate leakage between train and test"
print(f"[split] {drug}: train={len(train_files)} test={len(test_files)} isolates, overlap=0", flush=True)

model, tr_ds, va_ds, hist = ns["train_token"](
    drug=drug, gene=gene, train_files=train_files, train_labels=y_train,
    mode=mode, in_dim=in_dim, n_epochs=20, batch_size=8, lr=5e-4,
    out_root=args.out_root, compute_shap=False,
)

# Test dataset, built the same way as the training dataset.
test_label_map = dict(zip(test_files, y_test))
if drug in ns["multi_drugs"]:
    genes = ns["multi_drugs"][drug]
    test_ds = (ns["MultiGeneConcatDataset"](genes, drug, test_label_map) if mode == "full"
               else ns["PcaMultiGeneConcatDataset"](genes, drug, test_label_map, k=in_dim))
else:
    data_path = ns["embeddings_root"](gene, drug)
    if mode == "full":
        metas = [p for p in ns["glob"].glob(f"{data_path}/token/*_meta.npz") if "_pc" not in p]
        test_ds = ns["TokenMemmapMap"](metas, test_label_map)
    else:
        metas = ns["glob"].glob(f"{data_path}/token/PCA/*_pc{in_dim}_meta.npz")
        test_ds = ns["PcaMemmapMap"](metas, test_label_map, k=in_dim)

train_ids = set(tr_ds.dataset.ids) if hasattr(tr_ds, "dataset") and hasattr(tr_ds.dataset, "ids") else None
if train_ids is not None and hasattr(test_ds, "ids"):
    assert not train_ids & set(test_ds.ids), "test dataset overlaps training dataset"

test_auc = ns["evaluate_auc"](model, test_ds, device=ns["device"])
train_auc = ns["evaluate_auc"](model, tr_ds, device=ns["device"])
run_dir = Path(args.out_root) / "prediction" / f"{drug}_dim{in_dim}"
pd.DataFrame({"Drug": [drug], "Train_AUC": [train_auc], "Test_AUC": [test_auc],
              "n_test": [len(test_ds)]}).to_csv(run_dir / f"{drug}_train_test_auc.csv", index=False)
print(f"[done] {drug} {mode}: Train AUC={train_auc:.3f} | Test AUC={test_auc:.3f} (n_test={len(test_ds)})")
