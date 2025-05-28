import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from downstream_cnn_model import *
from torch.utils.data import DataLoader, TensorDataset
from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from utils.embed_gen_utils import *
from utils.classification_metric_utils import *
from utils.train_utils import *
import ipdb


def main(args):

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(n_gpu))


    # train_embed_path = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings_3000/zs_train_embeddings_phenotypes.npz"
    # test_embed_path = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings_3000/zs_val_embeddings_phenotypes.npz"

    # train_embed_path = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/embeddings_5000/dnabert2/zs_train_embeddings_phenotypes.npz"

    train_embed_path = os.path.join(args.saved_embed_dir, args.train_embed_name)
    test_embed_path = os.path.join(args.saved_embed_dir, args.val_embed_name)

    train_data = np.load(train_embed_path)
    train_embeddings = train_data['embeddings']
    train_labels = train_data['phenotypes']
    print("Loaded embeddings shape:", train_embeddings.shape) # Shape: (num_samples, num_genes, dim)
    print("Loaded phenotypes shape:", train_labels.shape) # Shape: (num_samples, num_drugs)

    test_data = np.load(test_embed_path)
    test_embeddings = test_data['embeddings']
    test_labels = test_data['phenotypes']
    print("Loaded test embeddings shape:", test_embeddings.shape) # Shape: (num_samples, num_genes, dim)
    print("Loaded test phenotypes shape:", test_labels.shape) # Shape: (num_samples, num_drugs)


    print("\nConsidering isolates with at least 1 resistance status across all drugs...")
    # obtain isolates with at least 1 resistance status to length of drugs
    num_drugs = train_labels.shape[-1]
    train_indices_with_R_phenotype = np.where(train_labels.sum(axis=1) != -num_drugs)[0]
    test_indices_with_R_phenotype = np.where(test_labels.sum(axis=1) != -num_drugs)[0]

    train_embeddings = train_embeddings[train_indices_with_R_phenotype, :, :]  # Select only the embeddings corresponding to the indices with resistance phenotype
    train_labels = train_labels[train_indices_with_R_phenotype, :]  # Select only the labels corresponding to the indices with resistance phenotype

    test_embeddings = test_embeddings[test_indices_with_R_phenotype, :, :]  # Select only the embeddings corresponding to the indices with resistance phenotype
    test_labels = test_labels[test_indices_with_R_phenotype, :]  # Select only the labels corresponding to the indices with resistance phenotype

    # Hyperparameters
    input_dim = train_embeddings.shape[-1]  # d  
    hidden_dim = 64                  # You can tune this
    output_dim = train_labels.shape[-1]     # 11 drugs
    batch_size = 128
    epochs = 10
    learning_rate = 5e-5
    weight_decay = 1e-5

    train_embeddings = torch.tensor(train_embeddings).permute(0, 2, 1)  # Make sure embeddings are a tensor, # Change shape to [batch_size, dim, num_drugs] for CNN input
    train_labels = torch.tensor(train_labels)          # Make sure labels are a tensor

    test_embeddings = torch.tensor(test_embeddings).permute(0, 2, 1)  # Make sure embeddings are a tensor, # Change shape to [batch_size, dim, num_drugs] for CNN input
    test_labels = torch.tensor(test_labels)          # Make sure labels are a tensor


    # Dataset and DataLoader
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_embeddings, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = MDCNN(dropout_rate=0)
    # model = SmallMDCNN()
    # model = SmallTransformer(input_dim=768, seq_len=12, num_heads=4, hidden_dim=256, num_layers=2, num_classes=11)
    # model = MDMLP(dropout_rate=0)
    # model = ShallowMDMLP(input_dim=12 * 768, num_classes=11, dropout_rate=0)
    # model = DynamicMDMLP(num_classes=11, dropout_rate=0, pooled_genes=12)
    model = model.cuda()

    # Threshold value for AUC
    auc_threshold = ThresholdValue()

    print("Loading trained model...")
    saved_path = os.path.join(args.saved_model_path, args.saved_model_name)
    model.load_state_dict(torch.load(saved_path, weights_only=True))

    print("Predicting on training data to get AUC thresholds...")
    y_train, y_train_pred = evaluate(model, train_dataloader, device)
    auc_thresholds, drug_to_threshold = calculate_auc_thresholds(y_train, y_train_pred, auc_threshold)

    print("\nEvaluating on test data...")
    y_test, y_test_pred = evaluate(model, test_dataloader, device)
    test_results = calculate_test_auc(y_test, y_test_pred, drug_to_threshold)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    test_results.to_csv(f"{args.output_path}/test_set_auc.csv")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train downstream resistance task model')
    parser.add_argument('--model_name', type=str, default="zhihan1996/DNABERT-S", help='Model name')
    parser.add_argument('--max_length', type=int, default=5000, help="Max length of tokens")
    parser.add_argument('--saved_embed_dir', type=str, default='/training_output/transfer_learn/embeddings', help="Saved embeds directory")
    parser.add_argument('--train_embed_name', type=str, default='zs_train_embeddings_phenotypes.npz', help="Embedding name")
    parser.add_argument('--val_embed_name', type=str, default='zs_val_embeddings_phenotypes.npz', help="Embedding name")
    parser.add_argument('--train_batch_size', type=int, default=128, help="Batch size used for training dataset")
    parser.add_argument('--val_batch_size', type=int, default=128, help="Batch size used for validating dataset")
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    # parser.add_argument('--datapath', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training', help="The dict of data")
    # parser.add_argument('--pkl_file', type=str, default='geno_pheno_full_combined.csv', help="Pickle file containing geno-pheno mapping of isolate strains per drug")
    # parser.add_argument('--train_dataname', type=str, default='geno_pheno_train_combined.csv', help="Name of the data used for training")
    # parser.add_argument('--val_dataname', type=str, default='geno_pheno_val_combined.csv', help="Name of the data used for validating")
    # parser.add_argument('--phenotype_file', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training/phenotype/master_table_resistance.csv', help="Contains the phenotype of the isolate strains per drug")
    # parser.add_argument('--genotype_input_directory', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training/genotype/combined/', help="Contains the genotype of the isolate strains per drug")

    parser.add_argument('--output_path', type=str, default='training_output/transfer_learn/classification_results', help="Directory to save the trained model")
    parser.add_argument('--saved_model_path', type=str, default='training_output/transfer_learn/saved_models', help="Directory to save the trained model")
    parser.add_argument('--saved_model_name', type=str, default='dnabert-mdcnn_cv_split_0.pt', help="Name of the saved model")
    
    args = parser.parse_args()
    main(args)
