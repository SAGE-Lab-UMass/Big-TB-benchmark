import os
import argparse
import numpy as np
from utils.ig_interpret_utils import *
from utils.model_utils import *
from utils.data_utils import *
from map_mar import get_confident_mutation_hits
from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from dataloader.locus_order import DRUGS

import ipdb



def main(args):
    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(n_gpu))

    torch.cuda.empty_cache()

    # Load model and tokenizer
    print("\nLoading dnabert and downstream model...")
    tokenizer, dnabert_s, downstream_model = get_models(args.dnabert_model, args.dnabert_model_max_len, args.downstream_model_name, args.downstream_model_path, device)
    print("done!\n")

    # Load data, use memory mapping!
    print("\nLoading test data...")
    test_dataloader = multi_gene_multi_drug_loader_csv(args, load_train=False, n_gpu=n_gpu)
    drug_index = DRUGS.index(args.drug)

    # Select a small representative subset of the data for integrated gradients
    test_subset_dataloader = select_data_subset_for_ig(test_dataloader, drug_index, max_samples=200, ratio_sensitive=0.2)


    # Ff token attributions already exist, load them
    global_token_attributions_path = os.path.join(args.output_path, f"global_token_attributions_{args.drug}.pt")
    if os.path.exists(global_token_attributions_path):
        print(f"Loading global token attributions from {global_token_attributions_path}")
        global_token_attributions = torch.load(global_token_attributions_path)
        print("done!\n")
    else:
        print(f"Token attributions not found, computing them from scratch for {args.drug}...\n")
        all_token_attributions, global_token_attributions = calculate_token_attributions(test_dataloader, dnabert_s, downstream_model, drug_index, tokenizer, args.dnabert_model_max_len)  # list of (batch_size, num_genes, seq_len)

        # Final stack: (total_samples, num_genes, seq_len)
        drug_all_token_attributions = torch.cat(all_token_attributions, dim=0)
        print(f"\nFinal token attributions shape: {drug_all_token_attributions.shape}")  # (total_samples, num_genes, seq_len)

        # Save the attributions to a file to avoid recomputing
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        torch.save(drug_all_token_attributions, os.path.join(args.output_path, f"all_token_attributions_{args.drug}.pt"))
        print(f"Saved all token attributions to {os.path.join(args.output_path, f'all_token_attributions_{args.drug}.pt')}\n")

        # Save the global attributions to a file
        torch.save(global_token_attributions, os.path.join(args.output_path, f"global_token_attributions_{args.drug}.pt"))
        print(f"Saved global token attributions to {os.path.join(args.output_path, f'global_token_attributions_{args.drug}.pt')}\n")

    
    # Top k most important tokens for the drug
    print("\nCalculating top k most important tokens for the drug...")
    top_token_pos_df = get_global_token_importance(global_token_attributions, top_k=100)

    # Map the token positions to the positions on the original sequence
    print("\nMapping token positions to original sequence...")
    token_seq_mapped_df = map_token_positions_to_original_sequence(tokenizer, top_token_pos_df)
    token_seq_mapped_df.to_csv(os.path.join(args.output_path, "mapped_top_token_positions.csv"), index=False)
    print(f"Mapped token positions saved to {os.path.join(args.output_path, 'mapped_top_token_positions.csv')}\n")

    # Calculate metrics for causal variant discovery
    print("\nCalculating metrics for causal variant discovery...")
    important_features = get_important_features_list(token_seq_mapped_df)

    get_confident_mutation_hits(
        args.vcf_who_map_dir, 
        important_features, 
        args.drug,
        model_embed_type="tokens",
        has_neg_strand=args.has_neg_strand,
        output_csv=f"confident_mutation_hits_{args.drug}.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train downstream resistance task model')
    parser.add_argument('--dnabert_model', type=str, default="zhihan1996/DNABERT-S", help='DNABERT-s Model name')
    parser.add_argument('--downstream_model_name', type=str, default="downstream_cnn.pt", help='Downstream Model name')
    parser.add_argument('--downstream_model_path', type=str, default="/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/saved_models", help='Path to the downstream model')
    parser.add_argument('--drug', type=str, default="RIFAMPICIN", help='Drug for which to compute importance features. The drugs should be in capital letters and should be present in the drug list') 


    parser.add_argument('--dnabert_model_max_len', type=int, default=5000, help="Max allowed length of tokens. ZS-5000, FT-1000")
    parser.add_argument('--train_batch_size', type=int, default=2, help="Train Batch size used for integrated gradients")
    parser.add_argument('--val_batch_size', type=int, default=2, help="Val Batch size used for integrated gradients")
    parser.add_argument('--test_model_dir', type=str, default="/root/trained_model", help='Directory to save trained models to test')
    parser.add_argument('--model_list', type=str, default="tnf, test", help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, test]')
    parser.add_argument('--data_dir', type=str, default="/root/data", help='Data directory')
    
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    parser.add_argument('--datapath', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/eval', help="The dict of data")
    parser.add_argument('--pkl_file', type=str, default='geno_pheno_full_combined.csv', help="Pickle file containing geno-pheno mapping of isolate strains per drug")
    parser.add_argument('--train_dataname', type=str, default='geno_pheno_train_combined.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='geno_pheno_val_combined.csv', help="Name of the data used for validating")
    parser.add_argument('--embed_dir', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings', help="Directory to save embeddings")
    parser.add_argument('--phenotype_file', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training/phenotype/master_table_resistance.csv', help="Contains the phenotype of the isolate strains per drug")
    parser.add_argument('--genotype_input_directory', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/eval/genotype/combined/', help="Contains the genotype of the isolate strains per drug")

    parser.add_argument('--output_path', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/interpretability/output/saved_attribution_scores', help="Directory to save the attribution scores per drug")
    parser.add_argument('--saved_model_path', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/saved_models', help="Directory to save the trained model")
    

    # Files needed for interpretability
    parser.add_argument('--vcf_who_map_dir', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/make_vcf-who_map/outputs/vcf_who_mapped_csv_combined', help="Directory to save the trained model")
    parser.add_argument('--has_neg_strand', type=bool, default=True, help="Whether the data has negative strand mutations. Default is False")
    
    args = parser.parse_args()
    main(args)