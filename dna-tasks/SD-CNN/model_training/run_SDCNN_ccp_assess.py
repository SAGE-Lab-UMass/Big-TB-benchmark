#!/usr/bin/env python
# coding: utf-8
"""
Single-Drug CNN - Test/Evaluation
--------------------------------------------
- Loads best model from cross-validation training
- Evaluates on held-out test set
- Computes AUC, sensitivity, specificity on test data
- Generates predictions for all test isolates
- Saves evaluation results and strain-level predictions
"""

import os, sys, yaml, sparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Import shared codebase
from tb_cnn_codebase import *
from parameters.locus_order import BASE_TO_COLUMN
import ipdb

def run():

    class MyCNN:
        """Wrapper for loading and predicting"""
        def __init__(self, model_path):
            self.model = models.load_model(
                model_path,
                custom_objects={
                    'masked_weighted_accuracy': masked_weighted_accuracy,
                    'masked_multi_weighted_bce': masked_multi_weighted_bce
                }
            )

        def predict(self, X_val):
            """
            Returns predicted probabilities for given X

            Parameters
            ----------
            X_val: np.ndarray or sparse matrix
                Features for prediction

            Returns
            -------
            np.ndarray: Predicted probabilities
            """
            return np.squeeze(self.model.predict(X_val))

    def compute_threshold_from_training(model, X_train, y_train, drug):
        """
        Compute optimal decision threshold from training set predictions

        Parameters
        ----------
        model: MyCNN
            Trained model with predict method
        X_train: np.ndarray
            Training feature matrix
        y_train: np.ndarray
            Training phenotype labels (0=R, 1=S, -1=missing)
        drug: str
            Drug name for logging

        Returns
        -------
        float: Optimal threshold value

        Raises
        ------
        Exception: If threshold computation fails
        """
        print(f"Making predictions on training set ({X_train.shape[0]} samples)...")
        y_train_pred = model.predict(X_train)
        
        print(f"Computing optimal threshold...")
        try:
            threshold_data = get_threshold_val(y_train.reshape(-1, 1), y_train_pred.reshape(-1, 1))
            threshold = float(threshold_data['threshold'])
            print(f"Computed threshold from training data: {threshold:.4f}")
            return threshold
        except Exception as e:
            print(f"ERROR: Failed to compute threshold: {e}")
            raise

    def load_or_compute_threshold(threshold_file, model, X_train, y_train, drug):
        """
        Load threshold from file or compute from training data

        Parameters
        ----------
        threshold_file: str or None
            Path to saved threshold file (CSV with 'threshold' column)
        model: MyCNN
            Trained model for computing threshold if needed
        X_train: np.ndarray
            Training feature matrix
        y_train: np.ndarray
            Training phenotype labels (0=R, 1=S, -1=missing)
        drug: str
            Drug name for logging

        Returns
        -------
        float: Decision threshold value

        Raises
        ------
        SystemExit: If critical error occurs
        """
        # Try loading from file first
        if threshold_file and os.path.isfile(threshold_file):
            try:
                threshold_df = pd.read_csv(threshold_file, index_col=0)
                threshold = float(threshold_df['threshold'].iloc[0])
                print(f"Loaded threshold from file: {threshold:.4f}")
                return threshold
            except Exception as e:
                print(f"Failed to load threshold from {threshold_file}: {e}")
                print(f"Computing threshold from training data instead...")
        else:
            if threshold_file:
                print(f"Threshold file not found at {threshold_file}")
            print(f"Computing threshold from training data instead...")
        
        # Compute from training data
        try:
            threshold = compute_threshold_from_training(model, X_train, y_train, drug)
            return threshold
        except Exception as e:
            print(f"CRITICAL: Could not compute threshold: {e}")
            sys.exit(1)

    def compute_test_metrics(y_true, y_pred, threshold):
        """
        Computes AUC, sensitivity, specificity for test set

        Parameters
        ----------
        y_true: np.ndarray
            Actual phenotype labels (0=R, 1=S, -1=missing)
        y_pred: np.ndarray
            Predicted probabilities (0-1)
        threshold: float
            Decision threshold for classification

        Returns
        -------
        pd.DataFrame: Single row with metrics for all valid samples
        """
        column_names = ['Algorithm', 'Drug', 'num_sensitive', 'num_resistant', 
                       'AUC', 'AUC_PR', 'threshold', 'spec', 'sens']
        
        # Filter to non-missing phenotypes
        non_missing_idx = np.where(y_true != -1)[0]
        
        if len(non_missing_idx) == 0:
            print(f" No valid phenotypes found for {DRUG}")
            return pd.DataFrame(
                [['SD-CNN', DRUG, 0, 0, np.nan, np.nan, threshold, np.nan, np.nan]],
                columns=column_names
            )
        
        y_valid = y_true[non_missing_idx].astype(int)
        y_pred_valid = y_pred[non_missing_idx]
        
        num_sensitive = np.sum(y_valid == 1)
        num_resistant = np.sum(y_valid == 0)
        
        # Need both classes for evaluation
        if num_sensitive == 0 or num_resistant == 0:
            print(f" Only one class present in test set (S={num_sensitive}, R={num_resistant})")
            return pd.DataFrame(
                [['SD-CNN', DRUG, num_sensitive, num_resistant, np.nan, np.nan, 
                  threshold, np.nan, np.nan]],
                columns=column_names
            )
        
        # Compute metrics
        auc = roc_auc_score(y_valid, y_pred_valid)
        auc_pr = average_precision_score(1 - y_valid, 1 - y_pred_valid)
        
        # Binarize predictions using threshold
        binary_pred = (y_pred_valid > threshold).astype(int)
        
        # Sensitivity = TP / (TP + FN) = correctly classified resistant (0)
        sensitivity = np.sum(np.logical_and(binary_pred == 0, y_valid == 0)) / num_resistant
        
        # Specificity = TN / (TN + FP) = correctly classified sensitive (1)
        specificity = np.sum(np.logical_and(binary_pred == 1, y_valid == 1)) / num_sensitive
        
        return pd.DataFrame(
            [['SD-CNN', DRUG, num_sensitive, num_resistant, auc, auc_pr, 
              threshold, specificity, sensitivity]],
            columns=column_names
        )

    # -------------------------
    # Load YAML parameters
    # -------------------------
    _, input_file = sys.argv
    kwargs = yaml.safe_load(open(input_file, "r"))
    output_path = kwargs["output_path"]
    X_sparse_path = kwargs["X_sparse_path"]
    saved_model_path = kwargs["saved_model_path"]
    DRUG = kwargs["drug"]

    # -------------------------
    # Load data
    # -------------------------
    parquet_file = kwargs["metadata_path"]
    h5_file = kwargs["h5_path"]

    if os.path.isfile(parquet_file) and os.path.isfile(h5_file):
        print("Found existing parquet/HDF5 files.")
    else:
        print("Creating genotype-phenotype dataset ...")
        make_geno_pheno_dataset(**kwargs)
        print("Done.")

    print("Loading combined geno+pheno DataFrame ...")
    df_geno_pheno = load_combined_geno_pheno(**kwargs)
    print(f"Loaded {len(df_geno_pheno)} isolates.\n")

    # -----------------------------------------------
    # Train/Test split (stratified by phenotype)
    # -----------------------------------------------
    df_geno_pheno = df_geno_pheno.reset_index(drop=True)
    all_idx = df_geno_pheno.index
    
    # Stratify by phenotype labels to balance class distribution
    y_for_stratification = df_geno_pheno[DRUG].values
    train_idx, test_idx = train_test_split(
        all_idx, 
        test_size=kwargs["test_size"], 
        random_state=kwargs["random_seed"],
        stratify=y_for_stratification
    )
    train_df = df_geno_pheno.loc[train_idx]
    test_df = df_geno_pheno.loc[test_idx]
    print(f"Training samples: {len(train_df)},  Test samples: {len(test_df)}")
    print(f"Training set phenotype distribution: {np.unique(y_for_stratification[train_idx], return_counts=True)}")
    print(f"Test set phenotype distribution: {np.unique(y_for_stratification[test_idx], return_counts=True)}")

    # -----------------------------------------------
    # Create or load X sparse features
    # -----------------------------------------------
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

    # Extract test data
    X_test = X_sparse[test_idx, :].todense()
    print(f"Test X shape: {X_test.shape}")

    # -----------------------------------------------
    # Extract and process test labels
    # -----------------------------------------------
    print("Encoding test phenotype labels ...")
    y_test_df, y_test_array = rs_encoding_to_numeric(test_df, DRUG)
    y_test = y_test_df.values.astype(int)
    y_test = y_test.reshape(-1, 1)
    print(f"Test y shape: {y_test.shape}")
    print(f"Test set phenotype distribution: {np.unique(y_test, return_counts=True)}")

    # -----------------------------------------------
    # Load best model and get threshold from training
    # -----------------------------------------------
    best_model_path = os.path.join(saved_model_path, "sd-cnn_model_best.h5")
    
    if not os.path.isfile(best_model_path):
        print(f"ERROR: Best model not found at {best_model_path}")
        print(f"Available models in {saved_model_path}:")
        if os.path.isdir(saved_model_path):
            for f in os.listdir(saved_model_path):
                print(f"      - {f}")
        sys.exit(1)
    
    print(f"Loading best model from: {best_model_path}")
    model = MyCNN(best_model_path)

    # -----------------------------------------------
    # Load or compute threshold
    # -----------------------------------------------
    threshold_file = kwargs.get("threshold_file", None)
    
    # Get training data for threshold computation if needed
    X_train = X_sparse[train_idx, :].todense()
    y_train_df, y_train_array = rs_encoding_to_numeric(train_df, DRUG)
    y_train = y_train_df.values.astype(int)
    y_train = y_train.reshape(-1, 1)
    
    # Load from file or compute from training predictions
    threshold = load_or_compute_threshold(threshold_file, model, X_train, y_train, DRUG)
    
    # Clean up training data
    del X_train, y_train_df, y_train_array, y_train, X_sparse

    # -----------------------------------------------
    # Evaluate on test set
    # -----------------------------------------------
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    # Compute test metrics
    test_results = compute_test_metrics(y_test.ravel(), y_pred, threshold)
    test_results_file = f"{output_path}_test_set_drug_auc.csv"
    test_results.to_csv(test_results_file, index=False)
    print(f"\nTest results saved to: {test_results_file}")
    print(test_results)

    # -----------------------------------------------
    # Save strain-level predictions
    # -----------------------------------------------
    # print("\n" + "="*70)
    # print("SAVING STRAIN-LEVEL PREDICTIONS")
    # print("="*70)
    
    # # Create prediction dataframe with test isolate identifiers
    # prediction_df = test_df[["index"]].reset_index(drop=True)
    # prediction_df[DRUG] = y_pred
    # prediction_df['phenotype_true'] = y_test.ravel()
    # prediction_df['binary_pred'] = (y_pred > threshold).astype(int)
    
    # strain_predictions_file = f"{output_path}_test_strain_predictions.csv"
    # prediction_df.to_csv(strain_predictions_file, index=False)
    # print(f"Strain predictions saved to: {strain_predictions_file}")
    # print(f"Predictions for {len(prediction_df)} test isolates")

    # -----------------------------------------------
    # Summary statistics
    # -----------------------------------------------
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Test set size: {len(test_df)} isolates")
    print(f"Algorithm: SD-CNN")
    print(f"Drug: {DRUG}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"\nMetrics:")
    for col in ['AUC', 'AUC_PR', 'spec', 'sens']:
        val = test_results[col].iloc[0]
        if not np.isnan(val):
            print(f"  {col:>10}: {val:.4f}")
        else:
            print(f"  {col:>10}: N/A")
    
    print(f"\nEvaluation complete!")
    print(f"Results: {test_results_file}")
    # print(f"Predictions: {strain_predictions_file}")


if __name__ == "__main__":
    run()