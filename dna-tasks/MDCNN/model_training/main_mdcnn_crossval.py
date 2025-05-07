#!/usr/bin/env python
# coding: utf-8
"""
Runs multitask model with conv-conv-pool architecture, 5 fold cross validation on training/validation set
This is the architecture used for the final MD-CNN model

Authors:
	Michael Chen (original version)
	Anna G. Green
	Chang Ho Yoon
"""

import sys
import glob
import os
import yaml
import ipdb
import time
import sparse
from datetime import datetime

import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from tb_cnn_codebase import *
from parameters.locus_order import drugs, BASE_TO_COLUMN

def run():

    def get_conv_nn():
        """
        Define convolutional neural network architecture

        NB filter_size is a global variable (int) given by the kwargs
        """

        kernel_size = len(BASE_TO_COLUMN)
        num_drugs = len(drugs)
        
		#TODO: replace X.shape with passed argument
        model = models.Sequential()
		#TODO: add filter size argument
        model.add(layers.Conv2D(
            64, (kernel_size, filter_size),
            data_format='channels_last',
            activation='relu',
            input_shape = X.shape[1:]
        ))
        model.add(layers.Lambda(lambda x: K.squeeze(x, 1)))
        model.add(layers.Conv1D(64, 12, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', name='d1'))
        model.add(layers.Dense(256, activation='relu', name='d2'))
        model.add(layers.Dense(13, activation='sigmoid', name='d4'))

        opt = Adam(learning_rate=np.exp(-1.0 * 9))

        model.compile(optimizer=opt,
                      loss=masked_multi_weighted_bce,
                      metrics=[masked_weighted_accuracy])

        return model

    class myCNN:
        """
        Class for handling CNN functionality

        """
        def __init__(self):
            self.model = get_conv_nn()
            self.epochs = N_epochs
            self.train_logs_dir = os.path.join(output_path, "train_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tensorboard_callback = TensorBoard(log_dir=self.train_logs_dir, histogram_freq=1)

        def fit_model(self, X_train, y_train, X_val=None, y_val=None):
            """
            X_train: np.ndarray
                n_strains x 5 (one-hot) x longest locus length x no. of loci
                Genotypes of isolates used for training
            y_train: np.ndarray
                Labels for isolates used for training

            X_val: np.ndarray (optional, default=None)
                Optional genotypes of isolates in validation set

            y_val: np.ndarray (optional, default=None)
                Optional labels for isolates in validation set

            Returns
            -------
            pd.DataFrame:
                training history (accuracy, loss, validation accuracy, and validation loss) per epoch

            """
            if X_val is not None and y_val is not None:
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    batch_size=128,
                    callbacks=[self.tensorboard_callback]
                )

                # TODO: Write history to a log file   
                # print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)
            else:
                history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128, callbacks=[tensorboard_callbacks])
                # print('\nhistory dict:', history.history)
                return pd.DataFrame.from_dict(data=history.history)

        def predict(self, X_val):
        
            return np.squeeze(self.model.predict(X_val))
        
        def save(self, saved_model_path):
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            
            return self.model.save(saved_model_path)

    _, input_file = sys.argv

    print("\nreading parameter file:")

    # load kwargs from config file (input_file)
    kwargs = yaml.safe_load(open(input_file, "r"))
    print(kwargs)
    print("\n")

    output_path = kwargs["output_path"]
    N_epochs = kwargs["N_epochs"]
    filter_size = kwargs["filter_size"]
    # pkl_file = kwargs["pkl_file"]
    pkl_file_sparse_train = kwargs['pkl_file_sparse_train']
    pkl_file_sparse_test = kwargs['pkl_file_sparse_test']
    saved_model_path = kwargs['saved_model_path']
    parquet_file = kwargs["metadata_path"]
    h5_file = kwargs["h5_path"]

    # train_indices_file = "train_indices.npy"
    # test_indices_file = "test_indices.npy"
    num_drugs = len(drugs)

    start = time.time()

    # Determine whether pickle already exists
    # if os.path.isfile(pkl_file):
    #     print("genotype-phenotype df pickle file already exists, proceeding with modeling")
    # else:
    #     print("creating genotype-phenotype df pickle")
    #     make_geno_pheno_pkl(**kwargs)
    #     print("done!\n")

    # # Get data from pickle
    # print("\nreading in the geno_pheno df pkl...")
    # df_geno_pheno = pd.read_pickle(pkl_file)
    # print("done!\n")


    
    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("genotype-phenotype df files already exist, proceeding with modeling")
    else:
        print("creating genotype-phenotype df dataset")
        make_geno_pheno_dataset(**kwargs)
        print("done!\n")

    print("loading combined genotype-phenotype data")
    df_geno_pheno = load_combined_geno_pheno(**kwargs)

    end = time.time()
    total_time = end - start
    print(f"Total time taken to read in the pkl file: {total_time/60} minutes\n")

    # Extract the directory to save the indices
    # directory = os.path.dirname(pkl_file)

    # # Create the full path for the .npy file
    # train_indices_file_path = os.path.join(directory, train_indices_file)
    # test_indices_file_path = os.path.join(directory, test_indices_file)

    # original split
    # train_indices = df_geno_pheno.query("category=='set1_original_10202'").index
    # test_indices = df_geno_pheno.query("category!='set1_original_10202'").index

    # Perform a 80/20 train-test split
    df_geno_pheno = df_geno_pheno.reset_index(drop=True)
    all_indices = df_geno_pheno.index
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    train_df = df_geno_pheno.loc[train_indices]
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of testing samples: {len(test_indices)}\n")

    if os.path.isfile(pkl_file_sparse_train) and os.path.isfile(pkl_file_sparse_test):
        print("X input already exists, loading X...")
        X_sparse_train = sparse.load_npz(pkl_file_sparse_train)

        # Load train and test indices if they are stored in files or regenerate them if possible
        # Assuming you may store indices as npz or in other format if they exist
        # if os.path.isfile(train_indices_file_path) and os.path.isfile(test_indices_file_path):
        #     print("loading train and test indices...")
        #     train_indices = np.load(train_indices_file_path)
        #     test_indices = np.load(test_indices_file_path)
        # Perform a 70/30 train-test split
        
    else:
        print("creating X from geno_pheno df...")
        X_all = create_X(df_geno_pheno)
        print("done!")

        X_sparse = sparse.COO(X_all)

        X_all = X_sparse.todense()
        assert (X_all.shape[0] == len(df_geno_pheno))
        
        print("\nsplitting the geno_pheno df into training and testing sets...")
        X_sparse_train = X_sparse[train_indices, :]
        X_sparse_test = X_sparse[test_indices, :]
        del X_sparse
        print("done!\n")

        print(f"saving X_train to {pkl_file_sparse_train} as compressed sparse matrix...")
        sparse.save_npz(pkl_file_sparse_train, X_sparse_train, compressed=False)
        print("done!\n")

        print(f"saving X_test to {pkl_file_sparse_train} as compressed sparse matrix...")
        sparse.save_npz(pkl_file_sparse_test, X_sparse_test, compressed=False)
        print("done!\n")

        del X_sparse_test

        # Save train and test indices for future use
        # np.save(train_indices_file_path, train_indices)
        # np.save(test_indices_file_path, test_indices)

    print("creating y from geno_pheno df...")
    # y_all_train, y_array = rs_encoding_to_numeric(df_geno_pheno.query("category=='set1_original_10202'"), drugs)
    y_all_train, y_array = rs_encoding_to_numeric(train_df, drugs)
    del train_df
    del train_indices
    del test_indices
    print("done!\n")
    

    # obtain phenotype data for CNN
    print("obtaining phenotype data for CNN...")
    y_all_train = y_all_train[drugs].values.astype(int)
    print("done!\n")

    print("considering isolates with at least 1 resistance status across all drugs...")
    # obtain isolates with at least 1 resistance status to length of drugs
    indices_with_R_phenotype = np.where(y_all_train.sum(axis=1) != -num_drugs)

    X = X_sparse_train[indices_with_R_phenotype]
    print(f"Original X shape is {X_sparse_train.shape}")
    print(f"filtered X shape is {X.shape}")

    y = y_all_train[indices_with_R_phenotype]
    print(f"Original y shape is {y_all_train.shape}")
    print(f"filtered y shape is {y.shape}")

    print(f"filtered data %age is {(y.shape[0] / y_all_train.shape[0])*100}%\n")

    alpha_matrix_path = kwargs["alpha_file"]
    alpha_matrix = load_alpha_matrix(alpha_matrix_path, y_array, df_geno_pheno)
    del df_geno_pheno
    del y_all_train
    del y_array
    del X_sparse_train
    print("done!\n")


    ### Perform 5-fold cross validation
    cv_splits = 5
    print(f"performing {cv_splits}-fold cross validation...")

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=1)

    column_names = ['Validation Split #', 'Algorithm', 'Drug', "num_sensitive", "num_resistant", 'AUC', 'AUC_PR', "Threshold", "Spec", "Sens"]
    results = pd.DataFrame(columns=column_names)
    i = 0

    for train_idx, (train, val) in enumerate(cv.split(X, y)):
        model = myCNN()
        X_train = X[train, :].todense()
        X_val = X[val, :].todense()
        y_train = y[train, :]
        y_val = y[val, :]

        print(f'\nfitting model for cv split {train_idx}..')
        history = model.fit_model(X_train, alpha_matrix[train, :], X_val, alpha_matrix[val, :])

        # Ensure the output_path directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(f'\nwriting cross validation history.. to {output_path}/history_cv_split{train_idx}.csv')
        history_file_path = os.path.join(output_path, f"history_cv_split{train_idx}.csv")
        history.to_csv(history_file_path)
        model.save(f"{saved_model_path}_cv_split_{train_idx}")

        print(f'\npredicting for cv split {train_idx}....')
        y_pred = model.predict(X_val)

        for idx, drug in enumerate(drugs):
            print(f"calculating metrics for drug: {drug}")
            non_missing_val = np.where(y_val[:, idx] != -1)[0]

            # Check if non_missing_val is empty (no valid data for this drug)
            if len(non_missing_val) == 0:
                # If no valid data, insert NaN values for metrics
                print(f"No valid data for drug: {drug} as all the rows are missing")
                results.loc[i] = [f'val_split{train_idx}', 'CNN', drug, 0, 0, np.nan, np.nan, val_threshold, np.nan, np.nan]
                i += 1
                continue  # Skip the rest of the loop and move to the next drug

            auc_y = np.reshape(y_val[non_missing_val, idx], (len(non_missing_val), 1))
            auc_preds = np.reshape(y_pred[non_missing_val, idx], (len(non_missing_val), 1))

            num_sensitive = np.sum(auc_y==1)
            num_resistant = np.sum(auc_y==0)

            # If we don't have at least 1 R and 1 S isolate we can't assess model
            if num_sensitive==0 or num_resistant==0:
                print(f"No resistant or sensitive isolates for drug: {drug}")
                results.loc[idx] = [f'val_split{train_idx}', 'CNN', drug, 0, 0, np.nan, np.nan, val_threshold, np.nan, np.nan]
                continue

            # Combine auc_y and auc_preds into a DataFrame
            data = np.hstack((auc_y, auc_preds))
            df = pd.DataFrame(data, columns=['auc_y', 'auc_preds'])


            val_auc = roc_auc_score(auc_y, auc_preds)
            val_auc_pr = average_precision_score(1 - y_val[non_missing_val, idx], 1 - y_pred[non_missing_val, idx])
            val__ = get_threshold_val(y_val[non_missing_val, idx], y_pred[non_missing_val, idx])
            val_threshold = float(val__["threshold"])
            val_spec = val__['spec']
            val_sens = val__['sens']

            results.loc[i] = [f'val_split{train_idx}', 'CNN', drug, num_sensitive, num_resistant, val_auc, val_auc_pr, val_threshold, val_spec, val_sens]

            i += 1

        print(f"\nwriting per cv split {train_idx} results to {output_path}/cv_split_{train_idx}_auc.csv")
        cv_split_auc_file_path = os.path.join(output_path, f"cv_split_{train_idx}_auc.csv")
        results.to_csv(cv_split_auc_file_path)

    K.clear_session()

    print(f"\nwriting results to {output_path}/auc.csv")
    auc_results_file_path = os.path.join(output_path, "auc.csv")
    results.to_csv(auc_results_file_path)


run()
