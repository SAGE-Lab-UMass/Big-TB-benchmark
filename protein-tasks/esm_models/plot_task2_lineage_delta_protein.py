"""plot_task2_lineage_delta_protein.py
Regenerates Figure 10 (fig:task2_lineage_delta_protein), matching the house style established for
Figure 8: per-architecture stacked subplots, red=decline/green=improve, per-bar value labels.
Per the caption: solid bar = Precision@10 change, hatched bar = Recall@10 change.
Built 2026-09-21. Underlying data (FINAL_controlled_shap_comparison.csv) unaffected by the LogReg fix.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

df = pd.read_csv("data/latest/lineage_ood_all_train/FINAL_controlled_shap_comparison.csv")
ARCH_MAP = {"regression": "LogReg", "cnn": "CNN", "transformer": "Transformer", "esm_full320": "ESM-320"}
df["architecture"] = df["architecture"].map(ARCH_MAP)
arch_order = ["LogReg", "CNN", "Transformer", "ESM-320"]
drugs = sorted(df["drug"].unique())  # 8 drugs, excludes amikacin/kanamycin per Table 6 scope

fig, axes = plt.subplots(len(arch_order), 1, figsize=(11, 13), sharex=True)
fig.suptitle("Task 2 Precision@10/Recall@10 change under lineage holdout relative to random split (Protein)",
             fontsize=13)

x = np.arange(len(drugs))
bar_w = 0.35

for ax, arch in zip(axes, arch_order):
    sub = df[df.architecture == arch].set_index("drug")
    p10 = np.array([sub.loc[d, "delta_p10"] if d in sub.index else 0 for d in drugs])
    r10 = np.array([sub.loc[d, "delta_r10"] if d in sub.index else 0 for d in drugs])

    colors_p = ["#2ca02c" if v >= 0 else "#d62728" for v in p10]
    colors_r = ["#2ca02c" if v >= 0 else "#d62728" for v in r10]

    ax.bar(x - bar_w / 2, p10, bar_w, color=colors_p, label="Precision@10")
    ax.bar(x + bar_w / 2, r10, bar_w, color=colors_r, hatch="//", edgecolor="white", label="Recall@10")

    for xi, v in zip(x - bar_w / 2, p10):
        ax.text(xi, v + (0.01 if v >= 0 else -0.01), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    for xi, v in zip(x + bar_w / 2, r10):
        ax.text(xi, v + (0.01 if v >= 0 else -0.01), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"$\\Delta$\n({arch})", fontsize=11, weight="bold")

legend_elems = [
    Patch(facecolor="#2ca02c", label="Improve"),
    Patch(facecolor="#d62728", label="Decline"),
    Patch(facecolor="white", edgecolor="black", label="solid = Precision@10"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="hatched = Recall@10"),
]
axes[0].legend(handles=legend_elems, loc="upper left", fontsize=8)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels([d.capitalize() for d in drugs], rotation=45, ha="right", fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("../latex_paper/images/task2_lineage_delta_protein.pdf")
print("[OK] Wrote task2_lineage_delta_protein.pdf (style-matched)")
