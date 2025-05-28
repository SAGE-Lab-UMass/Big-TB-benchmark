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
    # embeddings = np.load('/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings/train_embeddings_new.npy', mmap_mode='r')
    # labels = np.load('/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings/res_phenotypes_now.npy', mmap_mode='r')

    # print("embeddings shape:", embeddings.shape)

    # num_samples = embeddings.shape[0]
    # labels = labels[:num_samples]
    # print("labels shape:", labels.shape)



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

    # Load the embeddings and labels from the files
    # Load the embeddings and labels from the files
    # train_embed_path = os.path.join(args.saved_embed_dir, args.train_embed_name)
    # test_embed_path = os.path.join(args.saved_embed_dir, args.val_embed_name)

    # train_embed_path = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/transfer_learn/embeddings_3000/zs_train_embeddings_phenotypes.npz"

    train_embed_path = "/project/pi_annagreen_umass_edu/saishradha/project_data_curation/benchmarking/DNABERT_S/training_output/zero_shot/embeddings_5000/dnabert2/zs_train_embeddings_phenotypes.npz"


    train_data = np.load(train_embed_path)
    train_embeddings = train_data['embeddings']
    train_labels = train_data['phenotypes']

    print("Loaded embeddings shape:", train_embeddings.shape) # Shape: (num_samples, num_genes, dim)
    print("Loaded phenotypes shape:", train_labels.shape) # Shape: (num_samples, num_drugs)

    print("\nConsidering isolates with at least 1 resistance status across all drugs...")
    # obtain isolates with at least 1 resistance status to length of drugs
    num_drugs = train_labels.shape[-1]
    indices_with_R_phenotype = np.where(train_labels.sum(axis=1) != -num_drugs)[0]

    train_embeddings = train_embeddings[indices_with_R_phenotype, :, :]  # Select only the embeddings corresponding to the indices with resistance phenotype
    train_labels = train_labels[indices_with_R_phenotype, :]  # Select only the labels corresponding to the indices with resistance phenotype

    train_embeddings = torch.tensor(train_embeddings).permute(0, 2, 1)  # Make sure embeddings are a tensor, # Change shape to [batch_size, dim, num_drugs] for CNN input
    train_labels = torch.tensor(train_labels)          # Make sure labels are a tensor

    # Calculate min and max values
    min_value = train_embeddings.min().item()
    max_value = train_embeddings.max().item()
    mean_value = train_embeddings.mean().item()
    std_value = train_embeddings.std().item()

    print(f"Train Embeddings Min Value: {min_value:.6f}")
    print(f"Train Embeddings Max Value: {max_value:.6f}")
    print(f"Train Embeddings Mean Value: {mean_value:.6f}")
    print(f"Train Embeddings Std Dev: {std_value:.6f}")

    # Conditionally standardize the embeddings
    # train_embeddings = conditionally_standardize_embeddings(train_embeddings, std_threshold=0.2)

    # Check the new min-max values
    print(f"After Standardization - Min: {train_embeddings.min().item()}, Max: {train_embeddings.max().item()}")


    # TODO: first separate validation set
    # Dataset and DataLoader
    train_dataset = TensorDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Loss function
    criterion = MaskedMultiWeightedBCE()

    # def initialize_weights(m):
    #     if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight)

    # model.apply(initialize_weights)

    # Accuracy metric
    acc_metric = MaskedWeightedAccuracy()
    auc_threshold = ThresholdValue()

    print("Training model...")
    # trained_model = train(model, dataloader, optimizer, criterion, acc_metric, epochs=epochs, device=device)
    # trained_model, _ = train_kfold(model, dataset, optimizer, criterion, acc_metric, k_folds=5, epochs=epochs, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, device=device)
    trained_model = train_kfold_mod(train_dataset, drugs, criterion, args.learning_rate, args.weight_decay, acc_metric, auc_threshold, output_path=args.output_path, saved_model_path=args.saved_model_path, k_folds=5, epochs=args.num_epochs, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, random_seed=args.random_seed, device=device)


    

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
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train the model')
    parser.add_argument('--output_path', type=str, default='training_output/transfer_learn/classification_results', help="Directory to save the trained model")
    parser.add_argument('--saved_model_path', type=str, default='training_output/transfer_learn/saved_models', help="Directory to save the trained model")
    parser.add_argument('--random_seed', type=int, default=1, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    main(args)
