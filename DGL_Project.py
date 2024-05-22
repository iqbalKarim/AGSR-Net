import numpy as np
import pandas as pd
from MatrixVectorizer import MatrixVectorizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
from AGSR_Net.AGRNet import AGRNet
import random, torch

# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Check for CUDA (GPU support) and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    # Additional settings for ensuring reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

def preprocess_data(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Data cleansing: replace negative and NaN values with 0
    data = np.maximum(data, 0)
    data = np.nan_to_num(data)

    return data

def vectorize(data):
    # Vectorization (if needed)
    vectorizer = MatrixVectorizer()
    vectors = vectorizer.vectorize(data)

    return vectors

def anti_vectorize(data, size, diagonal=False):
    # Reverse the vectorization (if needed)
    graphs = np.zeros((data.shape[0], size, size))
    for idx, graph in enumerate(data):
        vectorizer = MatrixVectorizer()
        graphs[idx] = vectorizer.anti_vectorize(graph, size, diagonal)

    return graphs

lr_data_path = 'lr_train.csv'
hr_data_path = 'hr_train.csv'

# lr_matrix: N * 12720, hr_matrix: N * 35778
lr_matrix = preprocess_data(lr_data_path)
hr_matrix = preprocess_data(hr_data_path)

# Define a function to calculate statistics and return them in a dictionary
def calculate_statistics(data):
    statistics = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Standard Deviation': np.std(data),
        'Min': np.min(data),
        'Max': np.max(data)
    }
    return statistics

# Calculate statistics for LR and HR data
lr_stats = calculate_statistics(lr_matrix)
hr_stats = calculate_statistics(hr_matrix)

# Create a DataFrame to hold the statistics for comparison
df_stats = pd.DataFrame({'LR Data': lr_stats, 'HR Data': hr_stats})

# Round the numbers to four decimal places for better readability
df_stats = df_stats.round(4)

def plot_evaluation_metrics(fold_results):
    metrics = np.array(fold_results)

    # Calculate mean and standard deviation across folds for each metric
    metrics_mean = metrics.mean(axis=0)
    metrics_std = metrics.std(axis=0)

    # Define metric names
    metric_names = ['MAE', 'PCC', 'JSD', 'MAE-BC', 'MAE-EC', 'MAE-PR']

    # Determine the number of subplot rows and columns (up to 2 plots per line)
    n_folds = len(fold_results)
    n_rows = 2
    n_cols = (n_folds + 1) // n_rows + ((n_folds + 1) % n_rows > 0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharey=True)

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Plot each fold's metrics
    for i in range(n_folds):
        axs[i].bar(metric_names, metrics[i], color='skyblue')
        axs[i].set_title(f'Fold {i + 1}')

    # Adjust subplot index for the average metrics based on the number of folds
    avg_metrics_index = n_folds

    # Plot the average metrics with error bars on the next subplot
    axs[avg_metrics_index].bar(metric_names, metrics_mean, color='orange', yerr=metrics_std, capsize=5)
    axs[avg_metrics_index].set_title('Avg. Across Folds')

    # Adding error bars to the average metrics
    for i, (mean, std) in enumerate(zip(metrics_mean, metrics_std)):
        axs[avg_metrics_index].errorbar(metric_names[i], mean, yerr=std, fmt='k_', ecolor='black', capsize=5)

    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')

def create_graph_from_vector(vectorized_matrix, matrix_size):
  adjacency_matrix = MatrixVectorizer.anti_vectorize(vectorized_matrix, matrix_size)
  G = nx.from_numpy_array(adjacency_matrix)
  return G

def evaluate_predictions_vectorized(y_pred, y_true, matrix_size):
    """
    Evaluates the vectorized predictions against the ground truth.

    Parameters:
    - y_pred: Predictions, an n x d array where each row is a vectorized predicted matrix.
    - y_true: Ground truth, an n x d array where each row is a vectorized true matrix.
    - matrix_size: The size of one side of the square matrix before vectorization.
    """
    # Convert vectorized predictions and truths back to graphs
    pred_graphs = [create_graph_from_vector(y_pred[i], matrix_size) for i in range(y_pred.shape[0])]
    true_graphs = [create_graph_from_vector(y_true[i], matrix_size) for i in range(y_true.shape[0])]

    # Initialize lists to store centrality measures for all graphs
    betweenness_diffs, eigenvector_diffs, pagerank_diffs = [], [], []

    for pred_graph, true_graph in zip(pred_graphs, true_graphs):
        # Compute centrality measures for each graph
        betweenness_pred = nx.betweenness_centrality(pred_graph, weight="weight")
        betweenness_true = nx.betweenness_centrality(true_graph, weight="weight")
        eigenvector_pred = nx.eigenvector_centrality(pred_graph, max_iter=1000, tol=1e-06, weight="weight")
        eigenvector_true = nx.eigenvector_centrality(true_graph, max_iter=1000, tol=1e-06, weight="weight")
        pagerank_pred = nx.pagerank(pred_graph, weight="weight")
        pagerank_true = nx.pagerank(true_graph, weight="weight")

        # Compute differences in centrality measures
        betweenness_diffs.append(mean_absolute_error(list(betweenness_true.values()), list(betweenness_pred.values())))
        eigenvector_diffs.append(mean_absolute_error(list(eigenvector_true.values()), list(eigenvector_pred.values())))
        pagerank_diffs.append(mean_absolute_error(list(pagerank_true.values()), list(pagerank_pred.values())))

    # Average differences across all graphs
    avg_betweenness_diff = np.mean(betweenness_diffs)
    avg_eigenvector_diff = np.mean(eigenvector_diffs)
    avg_pagerank_diff = np.mean(pagerank_diffs)

    print(f"Average MAE Betweenness Centrality: {avg_betweenness_diff}")
    print(f"Average MAE Eigenvector Centrality: {avg_eigenvector_diff}")
    print(f"Average MAE PageRank: {avg_pagerank_diff}")

    # Compute standard evaluation metrics on flattened arrays for overall comparison
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    js_dis = jensenshannon(y_true.flatten(), y_pred.flatten())

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Pearson Correlation Coefficient (PCC): {pcc}")
    print(f"Jensen-Shannon Distance: {js_dis}")

    return mae, pcc, js_dis, avg_betweenness_diff, avg_eigenvector_diff, avg_pagerank_diff

# ################################################################################
# 3 Fold cross-validation
# N * 12720
X = np.array(lr_matrix)
# N * 35778
y = np.array(hr_matrix)
print(X.shape, y.shape)

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# training for AGRNet
fold_results = []
fold = 0

for train_index, test_index in kf.split(X):
    fold += 1
    print(f"Fold {fold}: ")
    X_train_tmp, X_test_tmp = X[train_index], X[test_index]
    y_train_tmp, y_test = y[train_index], y[test_index]

    # Initialize your GNN model here
    model = AGRNet()
    X_train = anti_vectorize(X_train_tmp, 160)

    X_test = anti_vectorize(X_test_tmp, 160)

    y_train = anti_vectorize(y_train_tmp, 268)

    y_test_graph = anti_vectorize(y_test, 268)
    
    print(y_train.shape)

    model.train(X_train, y_train, X_test, y_test_graph)

    # Predicting on the test set
    predictions_tmp = model.predict(X_test)
    print(predictions_tmp.shape)

    predictions = []
    for i in range(predictions_tmp.shape[0]):
        predictions.append(MatrixVectorizer.vectorize(predictions_tmp[i,:]))
    predictions = np.array(predictions)

    print(predictions.shape)
    # Serialize predictions
    predictions_serialized = predictions.flatten()

    # Create a DataFrame for the submission
    submission = pd.DataFrame({
        'ID': np.arange(1, len(predictions_serialized) + 1),
        'Predicted': predictions_serialized
    })

    # Save predictions to a CSV file
    submission.to_csv(f'predictions_fold_{fold}.csv', index=False)

    # Evaluate predictions
    mae, pcc, js_dis, betweenness_diff, eigenvector_diff, pagerank_diff = evaluate_predictions_vectorized(predictions, y_test, 268)

    # Store or print the fold results
    fold_results.append((mae, pcc, js_dis, betweenness_diff, eigenvector_diff, pagerank_diff))
    print()

# Calculate average metrics across folds
avg_mae = np.mean([result[0] for result in fold_results])
avg_pcc = np.mean([result[1] for result in fold_results])
avg_js_dis = np.mean([result[2] for result in fold_results])
avg_betweenness_diff = np.mean([result[3] for result in fold_results])
avg_eigenvector_diff = np.mean([result[4] for result in fold_results])
avg_pagerank_diff = np.mean([result[5] for result in fold_results])

print(f"\nAverage Metrics Across Folds - MAE: {avg_mae}, PCC: {avg_pcc}, JS Distance: {avg_js_dis}, Betweenness diff: {avg_betweenness_diff}, Eigenvector diff: {avg_eigenvector_diff}, Pagerank diff: {avg_pagerank_diff}")

plot_evaluation_metrics(fold_results)

# ################################################################################
# # Train final model using all data
# N * 12720
print('Now training using all the data')
X = np.array(lr_matrix)
# N * 35778
y = np.array(hr_matrix)
print(X.shape, y.shape)

model = AGRNet()
X_train = anti_vectorize(X, 160)
y_train = anti_vectorize(y, 268)

model.train(X_train, y_train, X_train, y_train)
################################################################################

test_data_path = 'lr_test.csv'

# lr_matrix: 112 * 12720
test_matrix_tmp = preprocess_data(test_data_path)

test_matrix = anti_vectorize(test_matrix_tmp, 160)

predictions_tmp = model.predict(test_matrix)

predictions = []
for i in range(predictions_tmp.shape[0]):
    predictions.append(MatrixVectorizer.vectorize(predictions_tmp[i,:]))
predictions = np.array(predictions)

# for kaggle submissions
predictions_flattened = predictions.flatten()

# Create an ID column with numbers starting from 1 up to the number of predictions
id_column = np.arange(1, predictions_flattened.size + 1)

# Assemble the data into a DataFrame
submission_df = pd.DataFrame({
    'ID': id_column,
    'Predicted': predictions_flattened
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)


