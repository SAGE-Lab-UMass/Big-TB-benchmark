import os
import sys
import yaml
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipdb

from sklearn.model_selection import train_test_split
from utils import create_output_dir
# from mutation_level_interpret_log_reg import get_confident_mutation_hits
from map_mar import get_confident_mutation_hits

shap.initjs()

def parse_arguments():
    """
    Parse command-line arguments.
    Expects a single argument: the path to the YAML parameter file.
    """
    if len(sys.argv) < 2:
        raise ValueError("Usage: python script.py <parameter_file.yaml>")
    return sys.argv[1]

def load_config(input_file):
    """
    Load and parse the YAML configuration file.
    Returns a dictionary with all required parameters.
    """
    print("\nReading parameter file:")
    print("Input file:", input_file)
    with open(input_file, "r") as f:
        config = yaml.safe_load(f)
    print(config)
    print("\n")
    return config

def prepare_output_dirs(config):
    """
    Prepare and save output locations using the create_output_dir utility.
    Returns the drug-specific output directory and the model path.
    """
    drug = config["drug"]
    model_dir = config["model_dir"]
    drug_output_dir, saved_model_path = create_output_dir(model_dir, drug)
    return drug_output_dir, saved_model_path

def read_genotypes(genotype_sites_file):
    """
    Read the genotypes of interest from CSV and construct genotype column names.
    """
    print("Reading in genotypes of interest...")
    genotypes_df = pd.read_csv(genotype_sites_file, index_col=0)
    genotype_columns = [f"{locus}_{site}" for locus, site in zip(genotypes_df.locus, genotypes_df.sites)]
    print("Done!\n")
    return genotype_columns

def load_input_data(input_data_file):
    """
    Load the main input data from CSV.
    """
    input_data_df = pd.read_csv(input_data_file, index_col=0, low_memory=False)
    return input_data_df

def split_data(input_data_df, test_size=0.3, random_state=42):
    """
    Perform a train-test split on the given DataFrame indices.
    Returns the train and test subsets of the original DataFrame.
    """
    all_indices = input_data_df.index
    train_indices, test_indices = train_test_split(
        all_indices, test_size=test_size, random_state=random_state
    )
    train_df = input_data_df.loc[train_indices]
    test_df = input_data_df.loc[test_indices]
    
    print("Total train samples:", train_df.shape)
    print("Total test samples:", test_df.shape)
    return train_df, test_df

def drop_missing_for_drug(train_df, drug, genotype_columns):
    """
    Drop missing values for the specified drug and extract the corresponding
    X (features) and Y (labels) DataFrames/Series for model fitting.
    """
    print(f"\nDropping missing values from training set for drug {drug}...")
    for_fitting = train_df.dropna(subset=[drug])
    X = for_fitting[genotype_columns]
    Y = for_fitting[drug]
    print("X_train_df shape after dropping NaN:", X.shape)
    print("y_train_df shape after dropping NaN:", Y.shape)
    print(f"\nNumber of samples for drug {drug}:")
    print(for_fitting.groupby(drug).size())
    return X, Y, for_fitting

def load_model(saved_model_path, drug):
    """
    Load the pre-trained Logistic Regression model for the specified drug.
    """
    print("Loading the model...")
    model_path = f"{saved_model_path}/LogisticRegression_bestC.model"
    classifier = joblib.load(model_path)
    return classifier

def compute_shap_values(classifier, X):
    """
    Compute SHAP values using the provided model and feature matrix X.
    Returns the SHAP values object.
    """
    print("Computing SHAP values...")
    explainer = shap.Explainer(classifier, X)
    shap_values = explainer(X)
    return shap_values

def plot_shap_summary(shap_values, drug, plot_dir, filename="shap_summary_plot.png"):
    """
    Generate and display the SHAP summary bar plot.
    """
    print("Generating SHAP summary bar plot...")
    shap_fig = shap.plots.bar(shap_values)

    # Create the directory if it doesn't exist
    output_plot_dir = os.path.join(plot_dir, drug)
    os.makedirs(output_plot_dir, exist_ok=True)
    output_path = os.path.join(output_plot_dir, filename)

    # Create the plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to: {output_path}")

    plt.close()


def get_feature_contributions(importance_df, importance_pec_threshold=0.2, features_below_threshold=10):
    shap_vals = importance_df["mean_abs_shap"].values
    feature_index = importance_df.index.values

    # Compute total importance once
    total_importance = shap_vals.sum()

    # Create a boolean mask for features exceeding the threshold
    mask = (shap_vals / total_importance) > importance_pec_threshold

    # 1) All features above threshold
    important_features = feature_index[mask].tolist()

    # 2) The top 10 from below threshold
    below_threshold_features = feature_index[~mask]
    ten_most_important_below_threshold = below_threshold_features[:features_below_threshold].tolist()

    return important_features, ten_most_important_below_threshold

def get_imp_features_by_summary(shap_values, genotype_columns, importance_pec_threshold=0.2):
    """
    Get the most important features by mean absolute SHAP value.
    Returns a DataFrame with the feature importance values.
    """
    shap_values_df = pd.DataFrame(shap_values.values, columns=genotype_columns)
    importance = np.abs(shap_values_df).mean(axis=0)
    ordered_columns = importance.sort_values(ascending=False).index.tolist()
    shap_values_df = shap_values_df[ordered_columns]

    importance_df = (
        pd.DataFrame(importance, columns=["mean_abs_shap"])
        .sort_values("mean_abs_shap", ascending=False)
    )
    important_features, ten_most_important_below_threshold = get_feature_contributions(importance_df, importance_pec_threshold)

    important_features = list(map(lambda x: x.strip("/"), important_features))
    ten_most_important_below_threshold = list(map(lambda x: x.strip("/"), ten_most_important_below_threshold))

    return important_features, ten_most_important_below_threshold


def get_important_features(drug, genotype_sites_file, input_data_file, saved_model_path, plot_dir):
    # 4) Read genotype columns
    genotype_columns = read_genotypes(genotype_sites_file)
    
    # 5) Load the main input data
    input_data_df = load_input_data(input_data_file)
    
    # 6) Split into train/test
    train_df, test_df = split_data(input_data_df, test_size=0.3, random_state=42)
    
    # 7) Drop missing values for the target drug and extract X, Y
    X_train, Y_train, for_fitting = drop_missing_for_drug(train_df, drug, genotype_columns)
    
    # 8) Load the pre-trained model
    classifier = load_model(saved_model_path, drug)
    
    # 9) Compute SHAP values on the training data
    shap_values = compute_shap_values(classifier, X_train)
    
    # 10) Get important mutations by SHAP summary
    important_features, ten_most_important_below_threshold = get_imp_features_by_summary(shap_values, genotype_columns)

    # plot SHAP figures
    plot_shap_summary(shap_values, drug, plot_dir)

    print(f"\nImportant features for drug {drug}:")
    print(important_features)

    print(f"\nTen most important features below threshold for drug {drug}:")
    print(ten_most_important_below_threshold)
    
    # Optional: Additional SHAP plots or model evaluation can be added here
    # shap.summary_plot(shap_values, X_train, plot_type="bar")
    # shap.dependence_plot("feature_name", shap_values, X_train)

    return important_features, ten_most_important_below_threshold

def main():
    # 1) Parse command-line arguments
    input_file = parse_arguments()
    
    # 2) Load configuration
    config = load_config(input_file)

    # 3) Prepare output directories
    drug_output_dir, saved_model_path = prepare_output_dirs(config)
    
    drug = config["drug"]
    genotype_sites_file = config["genotype_sites_file"]
    input_data_file = config["input_data_file"]
    output_dir = config["output_dir"]
    WHO_VCF_mapped_dir = config["WHO_VCF_mapped_dir"]
    has_neg_strand = bool(config["has_neg_strand"])
    model_type = config["model_type"]
    plot_dir = config["plot_dir"]
    output_file_name = f"confident_mutation_hits_{drug}.csv"

    important_features, ten_most_important_below_threshold = get_important_features(drug, genotype_sites_file, input_data_file, saved_model_path, plot_dir)

    get_confident_mutation_hits(
        WHO_VCF_mapped_dir, 
        important_features, 
        ten_most_important_below_threshold, 
        drug,
        model_type=model_type,
        has_neg_strand=has_neg_strand,
        output_csv=f"{output_dir}/{output_file_name}"
    )

    print("Most important features for {drug}: ", important_features)
    print("Most important below threshold features for {drug}: ", ten_most_important_below_threshold)



if __name__ == "__main__":
    main()
