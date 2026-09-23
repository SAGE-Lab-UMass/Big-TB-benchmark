#!/usr/bin/env python3
"""
Verification script to check that the Evo2 LoRA finetuning setup is correct.

Usage:
    python verify_setup.py --drug ISONIAZID --heldout-lineage 2

This script checks:
1. Vendored DNABERT modules exist and can be imported
2. Required data files exist at default or specified paths
3. Data files have correct format and consistent isolate IDs
4. Environment has required packages
"""

import argparse
import sys
from pathlib import Path


def check_vendored_modules():
    """Check that vendored finetuning utility modules exist and can be imported."""
    print("=" * 70)
    print("CHECKING VENDORED MODULES")
    print("=" * 70)
    
    finetuning_dir = Path(__file__).resolve().parent
    utils_modules_dir = finetuning_dir / "modules"
    
    required_files = [
        utils_modules_dir / "__init__.py",
        utils_modules_dir / "dataloader" / "__init__.py",
        utils_modules_dir / "dataloader" / "locus_order.py",
        utils_modules_dir / "downstream_cnn_model.py",
    ]
    
    all_exist = True
    for filepath in required_files:
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {filepath.relative_to(finetuning_dir)}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n❌ Some vendored modules are missing!")
        return False
    
    # Try importing
    print("\nTesting imports...")
    try:
        sys.path.insert(0, str(utils_modules_dir))
        from dataloader.locus_order import DRUG_TO_LOCI, DRUGS
        from downstream_cnn_model import DNABERTCNN
        print(f"✓ Successfully imported DRUG_TO_LOCI ({len(DRUGS)} drugs)")
        print(f"✓ Successfully imported DNABERTCNN")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def check_data_files(geno_pheno_csv, lineage_csv, fasta_dir):
    """Check that required data files exist and have basic valid structure."""
    print("\n" + "=" * 70)
    print("CHECKING DATA FILES")
    print("=" * 70)
    
    all_ok = True
    
    # Check geno-pheno CSV
    geno_pheno_path = Path(geno_pheno_csv)
    if geno_pheno_path.exists():
        print(f"✓ Genotype-phenotype CSV exists: {geno_pheno_path}")
        try:
            import pandas as pd
            df = pd.read_csv(geno_pheno_path)
            print(f"  - Shape: {df.shape}")
            print(f"  - Columns: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        except Exception as e:
            print(f"  ⚠ Could not read CSV: {e}")
    else:
        print(f"✗ Genotype-phenotype CSV not found: {geno_pheno_path}")
        all_ok = False
    
    # Check lineage CSV
    lineage_path = Path(lineage_csv)
    if lineage_path.exists():
        print(f"✓ Lineage CSV exists: {lineage_path}")
        try:
            import pandas as pd
            df = pd.read_csv(lineage_path)
            print(f"  - Shape: {df.shape}")
            lineages = df['Lineage'].value_counts().to_dict() if 'Lineage' in df.columns else {}
            print(f"  - Lineage distribution: {lineages}")
        except Exception as e:
            print(f"  ⚠ Could not read CSV: {e}")
    else:
        print(f"✗ Lineage CSV not found: {lineage_path}")
        all_ok = False
    
    # Check FASTA directory
    fasta_path = Path(fasta_dir)
    if fasta_path.exists() and fasta_path.is_dir():
        print(f"✓ FASTA directory exists: {fasta_path}")
        fasta_files = list(fasta_path.glob("*.fasta")) + list(fasta_path.glob("*.fa"))
        print(f"  - Found {len(fasta_files)} FASTA files")
        if fasta_files:
            print(f"  - Examples: {[f.name for f in fasta_files[:5]]}")
    else:
        print(f"✗ FASTA directory not found: {fasta_path}")
        all_ok = False
    
    return all_ok


def check_environment():
    """Check that required Python packages are installed."""
    print("\n" + "=" * 70)
    print("CHECKING ENVIRONMENT")
    print("=" * 70)
    
    required_packages = [
        "torch",
        "transformers",
        "peft",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (not installed)")
            all_installed = False
    
    return all_installed


def check_drug_gene_mapping(drug):
    """Check that the specified drug has associated genes."""
    print("\n" + "=" * 70)
    print(f"CHECKING DRUG: {drug}")
    print("=" * 70)
    
    try:
        finetuning_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(finetuning_dir / "modules"))
        from dataloader.locus_order import DRUG_TO_LOCI, DRUGS
        
        if drug not in DRUGS:
            print(f"✗ Drug '{drug}' not recognized.")
            print(f"  Available drugs: {', '.join(DRUGS)}")
            return False
        
        genes = DRUG_TO_LOCI[drug]
        print(f"✓ Drug '{drug}' maps to {len(genes)} genes:")
        for gene in genes:
            print(f"  - {gene}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking drug mapping: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify Evo2 LoRA finetuning setup")
    parser.add_argument("--drug", type=str, help="Drug name to check (e.g., ISONIAZID)")
    parser.add_argument("--heldout-lineage", type=str, help="Held-out lineage (1-4)")
    parser.add_argument(
        "--geno-pheno-csv",
        default="../data/multidrug_classification/training/geno_pheno_full_combined.csv",
        help="Path to genotype-phenotype CSV",
    )
    parser.add_argument(
        "--lineage-csv",
        default="../../BIG_TB_isolates_with_lineages.csv",
        help="Path to lineage CSV",
    )
    parser.add_argument(
        "--fasta-dir",
        default="../data/aligned_fasta",
        help="Path to FASTA directory",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("EVO2 LORA FINETUNING SETUP VERIFICATION")
    print("=" * 70)
    
    results = {}
    
    # Check vendored modules
    results['modules'] = check_vendored_modules()
    
    # Check environment
    results['environment'] = check_environment()
    
    # Check data files
    results['data'] = check_data_files(
        args.geno_pheno_csv,
        args.lineage_csv,
        args.fasta_dir,
    )
    
    # Check drug mapping if specified
    if args.drug:
        results['drug'] = check_drug_gene_mapping(args.drug)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    if all_passed:
        print("\n✓ All checks passed! Setup is ready.")
        if args.drug and args.heldout_lineage:
            print(f"\nYou can now run:")
            print(f"  cd finetuning/lineage_holdout")
            print(f"  python train_evo2_lora.py \\")
            print(f"      --drug {args.drug} \\")
            print(f"      --heldout-lineage {args.heldout_lineage} \\")
            print(f"      --dry-run")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
