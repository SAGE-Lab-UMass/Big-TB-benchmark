#!/usr/bin/env python
# coding: utf-8
"""
Single-Drug CNN 
-------------------------
- Uses parquet + HDF5 data format for genotype/phenotype storage
- Uses random train/test split (same as MD-CNN)
- Keeps single-drug prediction and masked loss
"""

import os, sys, yaml, sparse
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping


# Import shared codebase
from tb_cnn_codebase import *
from parameters.locus_order import BASE_TO_COLUMN
import ipdb

def run():

    def get_conv_nn(X, filter_size):
        """Define CNN architecture for single-drug prediction"""
        model = models.Sequential()
        model.add(layers.Conv2D(
            64, (5, filter_size),
            activation='relu',
            data_format='channels_last',
            input_shape=X.shape[1:]
        ))
        model.add(layers.Conv2D(64, (1, 12), activation='relu'))
        model.add(layers.MaxPooling2D((1, 3)))
        model.add(layers.Conv2D(32, (1, 3), activation='relu'))
        model.add(layers.Conv2D(32, (1, 3), activation='relu'))
        model.add(layers.MaxPooling2D((1, 3)))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        opt = Adam(learning_rate=np.exp(-9.0))
        model.compile(optimizer=opt,
                      loss=masked_multi_weighted_bce,
                      metrics=[masked_weighted_accuracy])
        return model

    class MyCNN:
        """Wrapper for training and predicting"""
        def __init__(self, X, filter_size, N_epochs):
            self.model = get_conv_nn(X, filter_size)
            self.epochs = N_epochs

        def fit_model(self, X_train, y_train, X_val=None, y_val=None, callbacks=None):
            if X_val is not None and y_val is not None:
                hist = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.epochs, 
                    batch_size=128,
                    callbacks=callbacks
                )
            else:
                hist = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=128)
            return pd.DataFrame(hist.history)

        def predict(self, X_val):
            """
            Returns
            -------
            predicted labels for given X
            """
            return np.squeeze(self.model.predict(X_val))
        
        def save(self, path, include_optimizer=True):
            """ saves the model """
            self.model.save(path, include_optimizer=include_optimizer)

    # -------------------------
    # Load YAML parameters
    # -------------------------
    _, input_file = sys.argv
    kwargs = yaml.safe_load(open(input_file, "r"))
    output_path     = kwargs["output_path"]
    X_sparse_path  = kwargs["X_sparse_path"]
    N_epochs        = kwargs["N_epochs"]
    filter_size     = kwargs["filter_size"]
    saved_model_path = kwargs["saved_model_path"]
    DRUG            = kwargs["drug"]

    # ----------------------------------------------------
    # Load data (parquet + HDF5 logic)
    # ----------------------------------------------------
    parquet_file = kwargs["metadata_path"]
    h5_file      = kwargs["h5_path"]

    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("Found existing parquet/HDF5 files.")
    else:
        print("Creating genotype-phenotype dataset ...")
        make_geno_pheno_dataset(**kwargs)
        print("Done.")

    print("Loading combined geno+pheno DataFrame ...")
    df_geno_pheno = load_combined_geno_pheno(**kwargs)
    print(f"Loaded {len(df_geno_pheno)} isolates.\n")

    # ---------------------------------------------------
    # Train/Test split
    # ---------------------------------------------------
    df_geno_pheno = df_geno_pheno.reset_index(drop=True)
    all_idx = df_geno_pheno.index
    train_idx, test_idx = train_test_split(
        all_idx, test_size=kwargs["test_size"], random_state=kwargs["random_seed"]
    )
    train_df = df_geno_pheno.loc[train_idx]
    test_df  = df_geno_pheno.loc[test_idx]
    print(f"Training samples: {len(train_df)},  Testing samples: {len(test_df)}")

    # ---------------------------------------------------
    # Create X (features) and y (labels)
    # ---------------------------------------------------
    if X_sparse_path and os.path.isfile(X_sparse_path):
        print(f"Loading existing X_sparse from: {X_sparse_path}")
        X_sparse = sparse.load_npz(X_sparse_path)
    else:
        try:
            print("Creating X array ...")
            X_all = create_X(df_geno_pheno, kwargs["drug"])
            X_sparse = sparse.COO(X_all)
            del X_all
            created_X = True
        except Exception as e:
            print(f"Failed to create X_sparse: {e}")
            X_sparse = None
            created_X = False

        # Save only if created successfully
        if created_X and X_sparse is not None:
            os.makedirs(os.path.dirname(X_sparse_path), exist_ok=True)
            sparse.save_npz(X_sparse_path, X_sparse)
            print(f"Saved X_sparse at: {X_sparse_path}")
        else:
            print("X_sparse was not created — skipping save.")

    X_train = X_sparse[train_idx, :].todense()
    X_test  = X_sparse[test_idx, :].todense()
    del X_sparse

    print(f"Train X shape: {X_train.shape}, Test X shape: {X_test.shape}")

    # -----------------------------------------------
    # Extract and process train labels
    # -----------------------------------------------
    print("Encoding train phenotype labels ...")
    y_df, y_array = rs_encoding_to_numeric(train_df, DRUG)
    y_train = y_df.values.astype(int)
    y_train = y_train.reshape(-1, 1)

    # ---------------------------------------------------
    # Alpha matrix (weights)
    # ---------------------------------------------------
    alpha_matrix_path = kwargs["alpha_file"]
    alpha_matrix = load_alpha_matrix(alpha_matrix_path, y_array, df_geno_pheno, **kwargs)
    alpha_matrix = alpha_matrix.reshape(-1, 1)
    del df_geno_pheno

    # Remove isolates with missing phenotypes (-1)
    mask_valid = (y_train != -1).ravel()

    X_train = X_train[mask_valid]
    y_train = y_train[mask_valid]
    alpha_matrix = alpha_matrix[mask_valid]

    print(f"Filtered training data: {X_train.shape[0]} isolates remaining after removing missing phenotypes.")
    print(f"Alpha matrix shape: {alpha_matrix.shape}")

    # print("Unique y values:", np.unique(y_train, return_counts=True))
    # print("alpha stats:", np.isnan(alpha_matrix).sum(), alpha_matrix.min(), alpha_matrix.max())
    # print("y stats:", np.isnan(y_train).sum(), np.unique(y_train, return_counts=True))


    # ---------------------------------------------------
    # Cross-validation training (with best model saving)
    # ---------------------------------------------------
    cv_splits = 5
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=1)
    results = pd.DataFrame(columns=['Algorithm', 'Drug', 'AUC', 'AUC_PR', 'threshold', 'spec', 'sens'])
    i = 0

    best_auc = -np.inf
    best_model_path = None

    # Ensure the full directory exists
    os.makedirs(saved_model_path, exist_ok=True)
    print(f"Model directory ready: {saved_model_path}")

    for fold, (train, val) in enumerate(kf.split(X_train, y_train)):
        print(f"\n Fold {fold+1}/{cv_splits}")

        # --- Train model ---
        model = MyCNN(X_train, filter_size, N_epochs)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1
        )

        history = model.fit_model(
            X_train[train], alpha_matrix[train],
            X_train[val], alpha_matrix[val],
            callbacks=[early_stop]
        )
        history_path = f"{output_path}_history_cv{fold}.csv"
        history.to_csv(history_path, index=False)
        print(f"Training history saved at: {history_path}")

        # --- Save trained model ---
        # model_path = os.path.join(saved_model_path, f"sd-cnn_model_cv{fold}.h5")
        # model.save(model_path, include_optimizer=True)
        # print(f" Model saved at: {model_path}")

        model_path = os.path.join(saved_model_path, f"cnn_model_cv{fold}.h5")
        model.save(model_path)
        print(f"Model saved at: {model_path}")


        # --- Validate model performance ---
        y_pred = model.predict(X_train[val])
        val_auc = roc_auc_score(y_train[val], y_pred)
        val_auc_pr = average_precision_score(1 - y_train[val], 1 - y_pred)
        val__ = get_threshold_val(y_train[val], y_pred)

        results.loc[i] = [
            'CNN', DRUG, val_auc, val_auc_pr,
            float(val__['threshold']), val__['spec'], val__['sens']
        ]
        i += 1

        # --- Track and save best model ---
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(saved_model_path, "sd-cnn_model_best.h5")
            model.save(best_model_path, include_optimizer=True)
            print(f"New best model (AUC={best_auc:.4f}) saved at: {best_model_path}")

        # --- Free GPU memory between folds ---
        K.clear_session()

    # --- Save overall results ---
    results_path = f"{output_path}_auc.csv"
    results.to_csv(results_path, index=False)
    print("\n Training complete.")
    print(f" Results saved at: {results_path}")
    if best_model_path:
        print(f" Best model: {best_model_path} (AUC={best_auc:.4f})")


if __name__ == "__main__":
    run()


