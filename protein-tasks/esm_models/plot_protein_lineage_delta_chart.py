"""plot_protein_lineage_delta_chart.py
Regenerates Figure 8 (fig:protein_lineage_delta_chart), matching the original per-architecture
stacked-subplot style (red=decline, green=improve, per-bar labels, -0.05 threshold line).
Rebuilt 2026-09-21 after correcting the LogReg (Ref-Alt) random-split baseline in Table 3.
"""
import pandas as pd
import matplotlib.pyplot as plt

DRUGS = ["rifampicin", "isoniazid", "ethambutol", "pyrazinamide", "streptomycin",
         "moxifloxacin", "ethionamide", "amikacin", "capreomycin"]
ARCHS = {"LogReg": "logreg_test_auc", "CNN": "cnn_test_auc",
         "Transformer": "transformer_test_auc", "ESM-320": "esm_full320_test_auc"}
LINEAGE_SOURCES = {
    "LogReg":      ("data/latest/lineage_ood_all_train/regression/{d}/all_lineage_summary.csv", "logreg"),
    "CNN":         ("data/latest/lineage_ood_all_train/cnn/{d}/all_lineage_summary.csv", None),
    "Transformer": ("data/latest/lineage_ood_all_train/transformer/{d}/all_lineage_summary.csv", None),
    "ESM-320":     ("data/latest/lineage_ood_all_train/esm/{d}/full_320/all_lineage_summary.csv", None),
}
THRESHOLD = -0.05

def lineage_mean(path, model_filter=None):
    d = pd.read_csv(path)
    d = d[d["feasible"] == True]
    if model_filter:
        d = d[d["model"] == model_filter]
    return d["auc"].mean()

table3 = pd.read_csv("data/latest/results/prediction/combined/Combined2_test_aucs_Table.csv").set_index("drug")

arch_order = ["LogReg", "CNN", "Transformer", "ESM-320"]
deltas = {}
for arch in arch_order:
    path_tpl, mf = LINEAGE_SOURCES[arch]
    deltas[arch] = {d: lineage_mean(path_tpl.format(d=d), mf) - table3.loc[d, ARCHS[arch]] for d in DRUGS}

pd.DataFrame(deltas).to_csv("data/latest/lineage_ood_all_train/protein_lineage_delta_chart_data.csv")

fig, axes = plt.subplots(len(arch_order), 1, figsize=(11, 12), sharex=True)
fig.suptitle("Lineage-holdout AUC change relative to random split (Protein models)", fontsize=14)

for ax, arch in zip(axes, arch_order):
    vals = [deltas[arch][d] for d in DRUGS]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    bars = ax.bar(range(len(DRUGS)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.008 if v >= 0 else -0.008), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(THRESHOLD, color="gray", linestyle="--", lw=1)
    ax.set_ylim(-0.25, 0.2)
    ax.set_ylabel(f"$\\Delta$AUC\n({arch})", fontsize=11, weight="bold")

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor="#d62728", label="Decline"), Patch(facecolor="#2ca02c", label="Improve"),
                plt.Line2D([0], [0], color="gray", linestyle="--", label=f"{THRESHOLD} (>5% drop) threshold")]
axes[0].legend(handles=legend_elems, loc="upper left", fontsize=9)

axes[-1].set_xticks(range(len(DRUGS)))
axes[-1].set_xticklabels([d.capitalize() for d in DRUGS], rotation=45, ha="right", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../latex_paper/images/protein_lineage_delta_chart.pdf")
print("[OK] Wrote protein_lineage_delta_chart.pdf (style-matched)")

for arch in arch_order:
    print(arch, {d: round(deltas[arch][d], 3) for d in DRUGS})
