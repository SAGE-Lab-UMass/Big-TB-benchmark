# Evo2 LoRA Lineage-Holdout Finetuning

This directory contains the Evo2 lineage-holdout LoRA finetuning workflow.

## Structure

- `train_evo2_lora.py`: Main end-to-end LoRA finetuning entrypoint.
- `train_lineage_holdout.py`: Zero-shot downstream lineage-holdout trainer.
- `tests/`: Verification and smoke-test scripts.
- `utils/`: Data-loading, lineage-split, and helper utilities.

## Quick Start

```bash
cd finetuning/lineage_holdout
python train_evo2_lora.py --drug ISONIAZID --heldout-lineage 2 --dry-run
```

## Tests

```bash
cd finetuning/lineage_holdout
sbatch tests/run_smoke_test.sh
sbatch tests/run_verification_suite.sh
```

## Related Docs

- `IMPLEMENTATION_SUMMARY.md`
- `LAYER20_CORRECTIONS.md`
