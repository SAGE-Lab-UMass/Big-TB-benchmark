import argparse
import os
from data.locus_coords import gene_coords
from msa.aligned_msa import run_aligned_pipeline
from msa.unaligned_msa import run_unaligned_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="Generate aligned or unaligned FASTAs for a specific gene from VCFs."
    )
    
    parser.add_argument("--gene", "-g", required=True, help="Target gene name (e.g., rpoB)")
    parser.add_argument("--group", "-gr", default="group_1", help="WHO resistance category (default: group_1)")
    parser.add_argument("--file-type", default="combined", choices=["combined", "normal", "cryptic"],
                        help="Output file type (default: combined)")
    parser.add_argument("--msa-type", choices=["aligned", "unaligned"], required=True,
                        help="Whether to generate aligned or unaligned FASTAs")
    
    
    # Data paths
    parser.add_argument("--genbank", default="data/genbank_reference_data/genome.gbff",
                        help="Path to the GenBank reference file")
    parser.add_argument("--input-vcf-paths", default="raw_vcf_paths/raw_vcf_normal_paths.txt",
                        help="Path to the input VCF paths file")
    parser.add_argument("--secondary-vcf-paths", default="raw_vcf_paths/raw_vcf_cryptic_paths.txt",
                        help="Path to the secondary VCF paths file")
    parser.add_argument("--output-dir", default="fastas/", help="Output base directory (default: fastas/)")

    args = parser.parse_args()

    # Look up gene coordinates from dictionary
    coords = gene_coords.get(args.group, {}).get(args.gene, [])
    if not coords:
        raise ValueError(f"Coordinates not found for gene '{args.gene}' in group '{args.group}'")

    start, end, sense = coords[0], coords[-2], coords[-1]

    # Define output path
    output_subdir = os.path.join(args.output_dir, args.msa_type)
    os.makedirs(output_subdir, exist_ok=True)
    output_filename = f"{args.gene}_{args.group}_{args.file_type}.fasta"
    output_path = os.path.join(output_subdir, output_filename)

    # Dispatch to the correct MSA pipeline
    if args.msa_type == "aligned":
        run_aligned_pipeline(
            input_vcf_paths=args.input_vcf_paths,
            start=start,
            end=end,
            sense=sense,
            output_path=output_path,
            genbank_path=args.genbank,
            gene=args.gene,
            group=args.group,
            secondary_vcf_paths=args.secondary_vcf_paths
        )
    else:  # unaligned
        run_unaligned_pipeline(
            input_vcf_paths=args.input_vcf_paths,
            start=start,
            end=end,
            sense=sense,
            output_path=output_path,
            genbank_path=args.genbank,
            gene=args.gene,
            group=args.group,
            secondary_vcf_paths=args.secondary_vcf_paths
        )

if __name__ == "__main__":
    main()
