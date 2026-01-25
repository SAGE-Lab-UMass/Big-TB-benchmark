import sys
import os
import joblib
import pandas as pd
import yaml
import ipdb

from scipy import stats
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import roc_auc_score

from tb_logreg_utils import get_threshold_val, set_parameters, create_output_dir
from parameters.locus_order import DRUG_TO_LOCI


# Input argument is parameter file
_, input_file = sys.argv

print("\nreading parameter file:")

# load kwargs from config file (input_file)
kwargs = yaml.safe_load(open(input_file, "r"))
print(kwargs)
print("\n")

drug = kwargs["drug"]
output_dir = kwargs["output_dir"]
max_iters = kwargs["max_iterations"]
penalty = kwargs["regularization"]
genotype_sites_file = kwargs["genotype_sites_file"]
input_data_file = kwargs["input_data_file"]


# Read in the genotypes of interest
print("reading in genotypes of interest")
genotypes = pd.read_csv(f"{genotype_sites_file}", index_col=0)

selected_loci = [f"/{gene}" for gene in DRUG_TO_LOCI[drug]]
drug_genotypes = genotypes[genotypes["locus"].isin(selected_loci)]


genotype_columns = [f"{x}_{y}" for x,y in zip(drug_genotypes.locus, drug_genotypes.sites)]
print("done!\n")

print(f"length of genotype columns for drug {drug}: {len(genotype_columns)}")



### Prepare the input data
input_data_df_old = pd.read_csv(f"{input_data_file}", index_col=0, low_memory=False)
# test_df = input_data_df.query("category!='set1_original_10202'")
# train_df = input_data_df.query("category=='set1_original_10202'")

# Retain only samples with non-missing values for the specified drug
input_data_df = input_data_df_old[input_data_df_old[drug].notna()].copy()

# number of columns in input data
print(f"total input samples with non-missing phenotypes for drug {drug}: {input_data_df.shape}")

# Perform a 80/20 train-test split
all_indices = input_data_df.index
train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42, stratify=input_data_df[drug])
train_df = input_data_df.loc[train_indices]
test_df = input_data_df.loc[test_indices]
print("Total train samples", train_df.shape)
print("Total test samples", test_df.shape)

print(f"\ndropping missing values from training set for drug {drug}")
for_fitting=train_df.dropna(subset=[drug])
X = for_fitting[genotype_columns]
Y = for_fitting[drug]
print("X_train_df shape after dropping NaN:", X.shape)
print("y_train_df shape after dropping NaN:", Y.shape)

print(f"\nnumber of samples for drug {drug}: {for_fitting.groupby(drug).size()}")

### Fit the GridSearchCV model to choose best C
parameters = set_parameters()
classifier = LogisticRegression(
    max_iter=max_iters,
    penalty=penalty,
    class_weight="balanced"
)
clf = GridSearchCV(classifier, parameters)
print("\nfitting the model...")
clf.fit(X, Y)
print(f"best parameters: {clf.best_params_}")

# Prepare and save output locations
drug_output_dir, saved_model_path = create_output_dir(output_dir, drug)
gridsearch_model_path = f"{saved_model_path}/GridSearchCV.model"
print(f"\nsaving grid search model to {gridsearch_model_path}...")
joblib.dump(clf, gridsearch_model_path)


### Run cross validation using the best C, assess accuracy, sensitivity, specificity
print(f"\nrunning cross validation of 5 splits...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
data = []
for train_index, test_index in kf.split(X.values):
    print(f"penalty: {penalty}")

    classifier = LogisticRegression(penalty=penalty, class_weight="balanced", max_iter=max_iters, **clf.best_params_)
    classifier.fit(X.values[train_index,:],Y.values[train_index])

    y_pred = classifier.predict_proba(X.values[test_index])[:,1]
    y_true = Y.values[test_index]

    cutoffs = get_threshold_val(y_true, y_pred)

    val_auc = roc_auc_score(y_true, y_pred)
    print(f"\nvalidation AUC {val_auc}")
    print(f"logreg cutoff threshold {cutoffs}")

    data.append([drug, val_auc, cutoffs['spec'], cutoffs['sens'], cutoffs['threshold']])

df = pd.DataFrame(data, columns=["drug", "AUC", "spec", "sens", "threshold"])
print(f"\nsaving cross validation results to {drug_output_dir}/XVal_accuracy.csv...")
df.to_csv(f"{drug_output_dir}/XVal_accuracy.csv")

### Fit LogisticRegression on best C, then assess on held-out set
print("chosen parameters", clf.best_params_)
classifer = LogisticRegression(penalty=penalty, class_weight="balanced", max_iter=max_iters, **clf.best_params_)
classifier.fit(X,Y)

print(f"saving best C model to {saved_model_path}/LogisticRegression_bestC.model...")
joblib.dump(classifier, f"{saved_model_path}/LogisticRegression_bestC.model")

print(f"\ndropping test samples with missing data for drug {drug}...")
test_df=test_df.dropna(subset=[drug])
print("total X_test samples with no missing data", test_df.shape)

X = test_df[genotype_columns]
Y = test_df[drug]

print("X_test_df shape after dropping NaN:", X.shape)
print("y_test_df shape after dropping NaN:", Y.shape)

print(f"\nrunning model on test set for drug {drug}...")
y_pred = classifier.predict_proba(X.values)[:,1]
y_true = Y.values

cutoffs = get_threshold_val(y_true, y_pred)
val_auc = roc_auc_score(y_true, y_pred)

# Save data
data=[drug, len(y_true), sum(y_true==1), sum(y_true==0), val_auc, cutoffs['spec'], cutoffs['sens'], cutoffs['threshold']]
df = pd.DataFrame([data], columns=["drug", "N", "N_S", "N_R", "AUC", "spec", "sens", "threshold"])

print(f"\nsaving test set results to {drug_output_dir}/test_set_accuracy.csv...")
df.to_csv(f"{drug_output_dir}/test_set_accuracy.csv")
