#%%
import numpy as np

def pairwise_comparison(gt_clusters, predicted_clusters):
    """
    Compare each pair of elements to determine if they are in the same or
    different clusters according to both ground truth and predictions,
    also calculating false positives and false negatives.

    Args:
    - gt_clusters (numpy array): Array where each element represents the ground truth cluster of that index.
    - predicted_clusters (numpy array): Array where each element represents the predicted cluster.

    Returns:
    - tp (int): Number of true positives.
    - tn (int): Number of true negatives.
    - fp (int): Number of false positives.
    - fn (int): Number of false negatives.
    """
    D = len(gt_clusters)
    tp = tn = fp = fn = 0
    for i in range(D):
        for j in range(i + 1, D):
            in_same_cluster_gt = gt_clusters[i] == gt_clusters[j]
            in_same_cluster_pred = predicted_clusters[i] == predicted_clusters[j]
            if in_same_cluster_gt and in_same_cluster_pred:
                tp += 1
            elif not in_same_cluster_gt and not in_same_cluster_pred:
                tn += 1
            elif not in_same_cluster_gt and in_same_cluster_pred:
                fp += 1
            elif in_same_cluster_gt and not in_same_cluster_pred:
                fn += 1
    return tp, tn, fp, fn

def compute_performance_metrics(tp, tn, fp, fn):
    """
    Compute accuracy, precision, recall, and F1 score based on performance metrics.

    Args:
    - tp (int): True positives.
    - tn (int): True negatives.
    - fp (int): False positives.
    - fn (int): False negatives.

    Returns:
    - accuracy (float): Clustering accuracy.
    - precision (float): Precision of clustering.
    - recall (float): Recall of clustering.
    - F1 (float): F1 score of clustering.
    """
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    fp_rate = fp / (fp + tn) if fp + tn > 0 else 0

    F1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, F1, fp_rate


def clustering_accuracy(gt_clusters, clusters):
    """
    Compute the clustering accuracy given the ground truth cluster sizes and the clusters array.

    Args:
    - cluster_sizes (list of int): List of cluster sizes, representing the ground truth.
    - clusters (numpy array): Array where each element's value represents the predicted cluster.

    Returns:
    - accuracy (float): Clustering accuracy.
    """
    # Construct the ground truth clusters array based on the cluster_sizes
    # Compute TP and TN
    tp, tn, fp, fn = pairwise_comparison(gt_clusters, clusters)
    # Compute accuracy
    accuracy, precision, recall, F1, fp_rate = compute_performance_metrics(tp, tn, fp, fn)
    return accuracy, precision, recall, F1, fp_rate, fp

def get_clusters_from_sizes(sizes):
    return np.concatenate([np.full(size, i) for i, size in enumerate(sizes)])


# %%

def random_test():
    # Parameters
    D = 2000  # Total number of elements

    # Generating random cluster sizes that sum to D
    np.random.seed(42)  # For reproducibility
    cluster_sizes = np.random.randint(1, 100, size=100)
    cluster_sizes = np.round(cluster_sizes / cluster_sizes.sum() * D).astype(int)
    # Adjust the last element to ensure the sum is exactly D
    cluster_sizes[-1] = D - cluster_sizes[:-1].sum()

    # Ensure no cluster size is 0 after adjustment
    cluster_sizes[cluster_sizes == 0] = 1
    cluster_sizes[-1] = D - cluster_sizes[:-1].sum()

    # Generating random clusters based on these sizes
    clusters = np.concatenate([np.full(size, i) for i, size in enumerate(cluster_sizes)])

    # Shuffle clusters to simulate prediction
    np.random.shuffle(clusters)

    # Test the clustering accuracy calculation with these random values
    #cluster_sizes, clusters[:10]  # Displaying the first 10 values for clusters for inspection

    clusters[0:21] = 0

    gt_clusters = get_clusters_from_sizes(cluster_sizes)
    accuracy, precision, recall, F1, fp_rate, fp = clustering_accuracy(gt_clusters, clusters)



if __name__ == "__main__":
    # Example usage:
    cluster_sizes = [3, 3, 4]  # For instance, 3 clusters with sizes 3, 3, and 4 respectively
    clusters = np.array([2, 2, 2, 1, 1, 1, 1, 0, 0, 0])  # Example predicted clusters

    gt_clusters = get_clusters_from_sizes(cluster_sizes)
    accuracy, precision, recall, F1, fp_rate, fp = clustering_accuracy(gt_clusters, clusters)
    print('Test matrix:')
    print(f'Accuracy: {accuracy},\nPrecision: {precision},\nRecall: {recall},\nF1: {F1},\nFalse positive rate: {fp_rate}')

    print('\n\nRandom matrix:')
    random_test()
# %%
