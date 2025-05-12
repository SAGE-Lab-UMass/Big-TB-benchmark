import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from downstream_cnn_model import MDCNN, MDMLP
from torch.utils.data import DataLoader, TensorDataset
from dataloader.dataloader import multi_gene_multi_drug_loader_csv
from utils.classification_metric_utils import *
from utils.train_utils import *
import ipdb


def main(args):
    # parser = argparse.ArgumentParser(description='Evaluate clustering')
    # args = parser.parse_args()

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(n_gpu))

    # get tokenizer and model
    # print("Loading tokenizer and model...")
    # tokenizer, _ = get_tokenizer_model(args.model_name, args.max_length)
    # print("done!\n")

    #-------------------

    # Use memory mapping!
    embeddings = np.load('/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings/train_embeddings_new.npy', mmap_mode='r')
    labels = np.load('/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings/res_phenotypes_now.npy', mmap_mode='r')

    print("embeddings shape:", embeddings.shape)

    num_samples = embeddings.shape[0]
    labels = labels[:num_samples]
    print("labels shape:", labels.shape)

    # current label shape is wrong, just fetched 20 samples, need 10000 samples, so using dataloader to fetch the correct set of labels
    # print("Loading train data...")
    # train_loader = multi_gene_multi_drug_loader_csv(args, load_train=True, n_gpu=n_gpu)
    # print("done!\n")

    # # get resistance phenotype labels
    # print("Getting resistance phenotype labels for our training data...")
    # train_res_phenotypes = calculate_phenotypes(train_loader, tokenizer, args.max_length, args.embed_dir)
    # print("done!\n")

    # num_samples = embeddings.shape[0]
    # res_phenotype_labels = train_res_phenotypes[:num_samples]
    # print("labels shape:", res_phenotype_labels.shape)


    # Hyperparameters
    input_dim = embeddings.shape[-1]  # d  
    hidden_dim = 64                  # You can tune this
    output_dim = labels.shape[-1]     # 11 drugs
    batch_size = 64
    epochs = 40
    learning_rate = 1e-4

    embeddings = torch.tensor(embeddings)  # Make sure embeddings are a tensor
    labels = torch.tensor(labels)          # Make sure labels are a tensor

    embeddings = embeddings.permute(0, 2, 1)  # Change shape to [batch_size, d, 11] for CNN input

    # TODO: first separate validation set
    # Dataset and DataLoader
    dataset = TensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MDCNN(dropout_rate=0)
    # model = MDMLP(dropout_rate=0)
    model = model.cuda()

    # Loss and Optimizer
    criterion = MaskedMultiWeightedBCE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Accuracy metric
    acc_metric = MaskedWeightedAccuracy()
    auc_threshold = ThresholdValue()

    print("Training model...")
    # trained_model = train(model, dataloader, optimizer, criterion, acc_metric, epochs=epochs, device=device)
    # trained_model, _ = train_kfold(model, dataset, optimizer, criterion, acc_metric, k_folds=5, epochs=epochs, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, device=device)
    trained_model = train_kfold_mod(model, dataset, drugs, optimizer, criterion, acc_metric, auc_threshold, output_path=args.output_path, saved_model_path=args.saved_model_path, k_folds=5, epochs=epochs, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, device=device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train downstream resistance task model')
    parser.add_argument('--test_model_dir', type=str, default="/root/trained_model", help='Directory to save trained models to test')
    parser.add_argument('--model_list', type=str, default="tnf, test", help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, test]')
    parser.add_argument('--data_dir', type=str, default="/root/data", help='Data directory')
    parser.add_argument('--model_name', type=str, default="zhihan1996/DNABERT-S", help='Model name')
    parser.add_argument('--max_length', type=int, default=5000, help="Max length of tokens")
    parser.add_argument('--train_batch_size', type=int, default=64, help="Batch size used for training dataset")
    parser.add_argument('--val_batch_size', type=int, default=64, help="Batch size used for validating dataset")
    parser.add_argument('--test_split', type=str, default=0.2, help="Test split ratio")
    parser.add_argument('--datapath', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training', help="The dict of data")
    parser.add_argument('--pkl_file', type=str, default='geno_pheno_full_combined.csv', help="Pickle file containing geno-pheno mapping of isolate strains per drug")
    parser.add_argument('--train_dataname', type=str, default='geno_pheno_train_combined.csv', help="Name of the data used for training")
    parser.add_argument('--val_dataname', type=str, default='geno_pheno_val_combined.csv', help="Name of the data used for validating")
    parser.add_argument('--embed_dir', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings', help="Directory to save embeddings")
    parser.add_argument('--phenotype_file', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training/phenotype/master_table_resistance.csv', help="Contains the phenotype of the isolate strains per drug")
    parser.add_argument('--genotype_input_directory', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/finetune_data/multidrug_classification/training/genotype/combined/', help="Contains the genotype of the isolate strains per drug")

    parser.add_argument('--output_path', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/classification_results', help="Directory to save the trained model")
    parser.add_argument('--saved_model_path', type=str, default='/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/saved_models', help="Directory to save the trained model")
    
    args = parser.parse_args()
    main(args)
