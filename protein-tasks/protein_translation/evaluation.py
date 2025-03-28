
from IPython.display import clear_output
from sklearn.metrics import (
    mean_absolute_error, r2_score, mean_squared_error, 
    median_absolute_error, roc_curve, auc, confusion_matrix, 
    f1_score, roc_auc_score, classification_report
)
import numpy as np
import matplotlib.pyplot as plt

def calculate_classification_metrics(y_test, y_pred, threshold):
    y_pred_binary = [1 if x >= threshold else 0 for x in y_pred]
    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    metrics = {
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'f1_score': f1_score(y_test, y_pred_binary),
        'roc_auc_score': roc_auc_score(y_test, y_pred)
    }
    return metrics

def calculate_regression_metrics(y_test, y_pred):
    metrics = {}
    metrics['r2'] = r2_score(y_test, y_pred)
    metrics['mae'] = mean_absolute_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['medae'] = median_absolute_error(y_test, y_pred)

    return metrics


def calculate_optimal_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]



# # Filter rows where the 'confidence' column contains "Assoc w R"
# filtered_df = filtered_df[filtered_df['confidence'].str.contains("Assoc w R", case=False, na=False)]

# # Extract the total unique amino acid positions after filtering
# total_actual_positives = len(np.unique(filtered_df['aa_pos']))

# print(f"Total actual positives (Assoc w R): {total_actual_positives}")

def compute_precision_recall(variants_df, gene_name, original_indices_above_cutoff):
    # Filter the DataFrame for the gene of interest
    filtered_df = variants_df[variants_df['gene'].str.contains(gene_name, case=False, na=False)]

    # Extract the amino acid position directly from the 'aa_pos' column
    filtered_df.dropna(subset=['aa_pos'], inplace=True)

    # Filter rows where the 'confidence' column contains "Assoc w R"
    filtered_df = filtered_df[filtered_df['confidence'].str.contains("Assoc w R", case=False, na=False)]
    # Debug: Print filtered data
    # print(f"Filtered data for gene {gene_name}:\n", filtered_df.head())

    # Count unique true positives
    true_positives = len(set(filtered_df['aa_pos']).intersection(set(original_indices_above_cutoff)))
    
    # Debug: Print true positives
    # print(f"True Positives for {gene_name}: {true_positives}")

    # Count unique predictions made by the model
    total_predictions = len(set(original_indices_above_cutoff))
    # print(f"Total Predictions for {gene_name}: {total_predictions}")

    # Precision: true positives / total predictions
    precision = true_positives / total_predictions if total_predictions > 0 else 0

    # Count the total actual positives in the data
    total_actual_positives = len(np.unique(filtered_df['aa_pos']))
    # print(f"Total Actual Positives for {gene_name}: {total_actual_positives}")

    # Recall: true positives / total actual positives
    recall = true_positives / total_actual_positives if total_actual_positives > 0 else 0
    # print("recall before percentage",recall)

    # Convert recall to percentage for better interpretability
    recall_percentage = recall * 100

    return precision*100, recall_percentage


def compute_feature_importance(model,original_indices,gene_name,percentile=99):

    cutoff_value = np.percentile(np.abs(model.coef_), percentile)
    # print("percentile cutoff value",cutoff_value)  
    # To get the top N features
    num_features_above_cutoff = np.sum(np.abs(model.coef_) >= cutoff_value)
    # Find indices of coefficients that are above the cutoff value
    indices_above_cutoff = np.where(np.abs(model.coef_) >= cutoff_value)[0]
    # print("Indices above cutoff in the reduced model:", indices_above_cutoff)

    # Map these indices to the original indices
    original_indices_above_cutoff = original_indices[indices_above_cutoff]
    # print("Corresponding original indices:", original_indices_above_cutoff)

    # Get the absolute coefficients for these indices and sort them in descending order
    coefficients_above_cutoff = np.abs(model.coef_[indices_above_cutoff])
    sorted_order = np.argsort(-coefficients_above_cutoff)  # Negative for descending order

    # Apply sorting to indices and coefficients
    sorted_indices_above_cutoff = indices_above_cutoff[sorted_order]
    sorted_original_indices_above_cutoff = original_indices_above_cutoff[sorted_order]
    sorted_coefficients = coefficients_above_cutoff[sorted_order]

    # Now we have the sorted coefficients and corresponding indices
    # print("Feature importance values and corresponding original indices in descending order:")
    # for coef, reduced_idx, original_idx in zip(sorted_coefficients, sorted_indices_above_cutoff, sorted_original_indices_above_cutoff):
    #     print(f"Coefficient: {coef:.4f}, Reduced Index: {reduced_idx}, Original Index: {original_idx}")


    # Print the total number of selected features for the specified gene
    # print(f"Total number of features selected by model for gene {gene_name}: {len(original_indices_above_cutoff)}")

    # plot_coefficients(model, original_indices, indices_above_cutoff, original_indices_above_cutoff, cutoff_value,gene_name)


    return sorted_original_indices_above_cutoff


def plot_loss_trajectory(loss_history, gene):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Trajectory')
    plt.grid(True)
    plt.savefig(f'{gene}_convergence-loss.png')
    plt.show()



def plot_coefficients_trajectory(coefficients_history, feature_indices, gene_name, valid_indices):
    # Map feature_indices to the positions within the subset of valid indices
    mapped_indices = [np.where(valid_indices == idx)[0][0] for idx in feature_indices if idx in valid_indices]
    
    plt.figure(figsize=(10, 6))
    for idx, mapped_idx in zip(feature_indices, mapped_indices):
        coef_values = [coef[mapped_idx] for coef in coefficients_history]
        plt.plot(coef_values, label=f'Feature {idx}')
    plt.title(f'Coefficient Trajectories for {gene_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{gene_name}_coefficient_trajectory.png')
    plt.show()

def plot_optimization_path_2d(fused_ridge, original_idx1, original_idx2, gene_name, valid_indices):
    # Map the original indices to valid indices within the coefficients array 
    if original_idx1 not in valid_indices or original_idx2 not in valid_indices:
        print(f"One or both of the indices ({original_idx1}, {original_idx2}) are not in valid indices.")
        return

    idx1 = np.where(valid_indices == original_idx1)[0][0]
    idx2 = np.where(valid_indices == original_idx2)[0][0]

    coef_history = np.array(fused_ridge.history['coefficients'])
    coef1_values = coef_history[:, idx1]
    coef2_values = coef_history[:, idx2]

    # Create a grid for the contour plot
    coef1_range = np.linspace(coef1_values.min(), coef1_values.max(), 100)
    coef2_range = np.linspace(coef2_values.min(), coef2_values.max(), 100)
    coef1_grid, coef2_grid = np.meshgrid(coef1_range, coef2_range)

    # Compute the objective function values on the grid
    Z = np.zeros_like(coef1_grid)
    for i in range(coef1_grid.shape[0]):
        for j in range(coef1_grid.shape[1]):
            coef = fused_ridge.coef_.copy()
            coef[idx1] = coef1_grid[i, j]
            coef[idx2] = coef2_grid[i, j]
            Z[i, j] = fused_ridge._custom_loss(coef, fused_ridge.X, fused_ridge.y)

    # Plot the contour
    plt.figure(figsize=(8, 6))
    plt.contourf(coef1_grid, coef2_grid, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Objective Function Value')
    plt.plot(coef1_values, coef2_values, 'w.-', markersize=4)
    plt.title(f'Optimization Path for Coefficients {original_idx1} and {original_idx2} ({gene_name})')
    plt.xlabel(f'Coefficient {original_idx1}')
    plt.ylabel(f'Coefficient {original_idx2}')
    plt.grid(True)
    plt.savefig(f'{gene_name}_optimization_trajectory.png')
    plt.show()





def find_best_percentile(model, original_indices, gene_name, variants_df, percentiles, compute_feature_importance, compute_precision_recall):
    best_percentile = None
    best_precision = 0
    best_recall = 0
    best_f1_score = 0
    top_features = []

    for percentile in percentiles:
        important_positions = compute_feature_importance(model, original_indices, gene_name, percentile)
        precision, recall = compute_precision_recall(variants_df, gene_name, important_positions)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero

        if f1 > best_f1_score:
            best_percentile = percentile
            best_f1_score = f1
            best_precision = precision
            best_recall = recall
            top_features = important_positions[:10]

    return {
        'best_percentile': best_percentile,
        'best_f1_score': best_f1_score,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'top_features': top_features
    }




fys = []  # Initialize fys as an empty list
def plot_function(xdims, ydims, f, title, xstar, X, y, idx1, idx2):
    global fys
    gx1, gx2 = np.meshgrid(
        np.arange(xdims[0], xdims[1], (xdims[1] - xdims[0]) / 50.0),
        np.arange(ydims[0], ydims[1], (ydims[1] - ydims[0]) / 50.0)
    )
    gx1l = gx1.flatten()
    gx2l = gx2.flatten()
    Z = np.zeros_like(gx1l)

    # Calculate objective function for each grid point
    for i in range(len(gx1l)):
        coef = xstar.copy()
        coef[idx1] = gx1l[i]
        coef[idx2] = gx2l[i]
        Z[i] = f(coef, X, y)  # Passing X, y to `_custom_loss`

    Z = Z.reshape(gx1.shape)

    plt.contourf(gx1, gx2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Objective Function Value')
    plt.title(title)
    plt.xlabel(f'Coefficient {idx1}')
    plt.ylabel(f'Coefficient {idx2}')
    plt.grid(True)


def plot_step(x, xprime, alpha):
    plt.plot([x[0], xprime[0]], [x[1], xprime[1]], '-w', alpha=alpha)

def plot_all_steps(xs):
    for i in range(xs.shape[0] - 1):
        plot_step(xs[i, :], xs[i + 1, :], alpha=0.7)
    plt.plot(xs[-1, 0], xs[-1, 1], 'ow')




def optimization_plots(fused_ridge, original_idx1, original_idx2, gene_name, valid_indices):
    # Determine xstar as the final coefficients after convergence
    xstar = fused_ridge.coef_
    
    # Map original indices to valid indices within the coefficients array
    if original_idx1 not in valid_indices or original_idx2 not in valid_indices:
        print(f"One or both of the indices ({original_idx1}, {original_idx2}) are not in valid indices.")
        return

    idx1 = np.where(valid_indices == original_idx1)[0][0]
    idx2 = np.where(valid_indices == original_idx2)[0][0]

    coef_history = np.array(fused_ridge.history['coefficients'])
    coef1_values = coef_history[:, idx1]
    coef2_values = coef_history[:, idx2]

    # Define range around xstar for contour plotting
    coef1_range = np.linspace(coef1_values.min(), coef1_values.max(), 100)
    coef2_range = np.linspace(coef2_values.min(), coef2_values.max(), 100)
    coef1_grid, coef2_grid = np.meshgrid(coef1_range, coef2_range)

    # Calculate the objective function values for the contour plot
    Z = np.zeros_like(coef1_grid)
    for i in range(coef1_grid.shape[0]):
        for j in range(coef1_grid.shape[1]):
            coef = xstar.copy()  # Start from the converged solution
            coef[idx1] = coef1_grid[i, j]
            coef[idx2] = coef2_grid[i, j]
            Z[i, j] = fused_ridge._custom_loss(coef, fused_ridge.X, fused_ridge.y)

    # Plot the contour
    plt.figure(figsize=(16, 6))
    
    # Panel 1: Full Contour and Path
    plt.subplot(1, 3, 1)
    plt.contourf(coef1_grid, coef2_grid, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Objective Function Value')
    plt.plot(coef1_values, coef2_values, 'w.-', markersize=4)
    plt.plot(xstar[idx1], xstar[idx2], 'r*', markersize=10, label="Converged xstar")
    plt.legend()
    plt.title(f'Optimization Path for Coefficients {original_idx1} and {original_idx2} ({gene_name})')
    plt.xlabel(f'Coefficient {original_idx1}')
    plt.ylabel(f'Coefficient {original_idx2}')
    # Define buffer size
    buffer = 0.01 * (coef1_values.max() - coef1_values.min())

    # Set x and y limits with a buffer around the trajectory range
    plt.xlim(coef1_values.min() - buffer, coef1_values.max() + buffer)
    plt.ylim(coef2_values.min() - buffer, coef2_values.max() + buffer)
    plt.grid(True)

    # Panel 2: Zoomed Path near Final Iterations
    plt.subplot(1, 3, 2)
    # zoom_factor = 0.05  # Adjust zoom level here to increase visibility
    zoom_factor_x = 0.02  # Customize these values as per the convergence pattern
    zoom_factor_y = 0.02
    x1_center, x2_center = xstar[idx1], xstar[idx2]
    plt.xlim(x1_center - zoom_factor_x, x1_center + zoom_factor_x)
    plt.ylim(x2_center - zoom_factor_y, x2_center + zoom_factor_y)
    plt.contourf(coef1_grid, coef2_grid, Z, levels=50, cmap='viridis')
    plt.plot(coef1_values, coef2_values, 'w.-', markersize=4)
    plt.plot(xstar[idx1], xstar[idx2], 'r*', markersize=10)
    plt.title("Zoomed Optimization Path")
    plt.xlabel(f"Coefficient {original_idx1}")
    plt.ylabel(f"Coefficient {original_idx2}")
    plt.grid(True)
    
    # Panel 3: Objective Value vs. Iteration
    plt.subplot(1, 3, 3)
    plt.plot(fused_ridge.history['loss'], 'k-')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function')
    plt.title("Objective vs Iteration")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{gene_name}_optimization_plots.png')
    plt.show()


def spectra_SPC(df_results,model_name,gene_name):
    # Convert results to a DataFrame
    # df_results = pd.DataFrame(performance_results)

    # Compute Mean Performance for Each Lambda Value
    mean_performance = df_results.groupby("lambda")["pr_auc"].mean()

    # Plot Spectral Performance Curve (SPC)
    plt.figure(figsize=(8, 5))
    plt.plot(mean_performance.index, mean_performance.values, marker="o", linestyle="-", label="Log Performance")
    plt.xlabel("Spectral Parameter (Lambda)")
    plt.ylabel("Mean PR-AUC")
    plt.title("Spectral Performance Curve (SPC) for Log Regression: RIF")
    plt.legend()
    plt.grid()
    plt.savefig(f'{gene_name}_spectra_spc_{model_name}.png')
    plt.show()