#%%
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, SpectralClustering
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import json
import jax
import os

from model.model import Maskformer, SensorAggregator, default_mask_model_settings
from evaluation.clustering_accuracy_metrics import clustering_accuracy, get_clusters_from_sizes
from dataloaders.pendulum_dataloader import get_default_pendulum_data

def our_model_adjacency(model_path, dataset_name=None):
    # chec if settings file exists
    settings_file = model_path + '-settings.json'
    if not os.path.exists(settings_file):
        settings = default_mask_model_settings(dataset_name, jax.random.PRNGKey(0))
    else:
        settings = json.load(open(model_path + '-settings.json'))
        settings['key'] = jax.random.PRNGKey(0)
    if 'disable_adjacency' not in settings:
        settings['disable_adjacency'] = False
    if 'rank_div' not in settings:
        settings['rank_div'] = 1
    if 'instance_norm' not in settings:
        settings['instance_norm'] = False
    if 'revin' not in settings:
        settings['revin'] = False
    model = Maskformer(**settings)
    model = eqx.tree_deserialise_leaves(model_path, model)

    adjacency = SensorAggregator.norm_adjacency(model.sensor_aggregator.__wrapped__.static_attention)
    return adjacency

def threshold_clusters_from_adjacency(adj_matrix, threshold):
    thresholded_adj = adj_matrix > threshold
    csr_adj = csr_matrix(thresholded_adj)
    n_components, labels = connected_components(csgraph=csr_adj, directed=False)
    return labels

def generate_block_matrix(cluster_sizes):
    blocks = [np.ones((size, size)) for size in cluster_sizes]
    block_matrix = np.block([[blocks[i] if i == j else np.zeros((blocks[i].shape[0], blocks[j].shape[1]))
                              for j in range(len(blocks))] for i in range(len(blocks))])

    return block_matrix - np.diag(np.diag(block_matrix))

def generate_random_matrix(size, density=0.1):
    random_matrix = np.random.rand(size, size) < density  # Generate a sparse matrix
    random_matrix = np.random.rand(size, size) * random_matrix  # Apply random weights to connections
    random_matrix = (random_matrix + random_matrix.T) / 2  # Make it symmetric
    np.fill_diagonal(random_matrix, 0)  # Optional: Remove self-connections
    return random_matrix

def laplacian(adjacency_matrix, normed=True):
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    # normalize
    if normed:
        laplacian_matrix = np.diag(1.0 / np.sqrt(degree_matrix.diagonal())) @ laplacian_matrix @ np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    return laplacian_matrix

def laplacian_eigenvalues(adjacency_matrix):
    L = laplacian(adjacency_matrix, normed=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Sort the eigenvalues in ascending order, if not already
    idxs = eigenvalues.argsort()
    sorted_eigenvalues  = eigenvalues[idxs]
    print(eigenvectors.shape)
    sorted_eigenvectors = eigenvectors[:, idxs]
    return sorted_eigenvalues, sorted_eigenvectors

def plot_eigengaps(eigenvalues, idx=None):
    max_eigvals = 2500
    gaps = np.diff(eigenvalues)[1:max_eigvals]
    if idx is None:
        gaps = np.diff(eigenvalues)[:max_eigvals]
        idx = np.argmax(gaps)
    x = np.arange(len(gaps[:max_eigvals]))
    y = gaps[:max_eigvals]
    fig = plt.figure()
    plt.plot(x, y)
    plt.scatter(x[idx], gaps[idx], color='red', s=100, zorder=5, label='Knee Point')
    plt.title(f'Eigenvalue gaps, {len(gaps)} gaps, {idx+1} clusters')
    plt.show()

    return idx

def elbow_cluster_count_from_eigvals(eigenvalues):
    #x = sorted_eigenvalues[:len(gaps)]
    gaps = np.diff(eigenvalues[0:])
    x = np.arange(len(gaps))
    y = gaps

    # Create the line equation between the first and last points
    start_point = np.array([x[0], y[0]])
    end_point = np.array([x[-1], y[-1]])
    line_vec = end_point - start_point
    line_unitvec = line_vec / np.linalg.norm(line_vec)

    # Compute the projection lengths of each point onto the line
    point_vecs = np.column_stack((x, y)) - start_point
    proj_lengths = np.dot(point_vecs, line_unitvec)

    # Find the perpendicular distances from the points to the line
    proj_points = np.outer(proj_lengths, line_unitvec) + start_point
    perp_distances = np.linalg.norm(point_vecs - proj_points, axis=1)

    # The knee is the point with the maximum perpendicular distance
    knee_index = np.argmax(perp_distances)
    knee_value = y[knee_index]

    n_clusters = knee_index + 1

    plot_eigengaps(eigenvalues, knee_index)

    return n_clusters, knee_value, x, y

def library_spectral_clustering(adjacency, k):
    #sc = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=100, assign_labels='discretize')
    #sc = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=100, assign_labels='kmeans')
    sc = SpectralClustering(n_clusters=k, affinity='precomputed', n_init=100, assign_labels='kmeans')
    labels = sc.fit_predict(adjacency)
    return labels


def spectral_clustering(adjacency, k):
    eigenvalues, eigenvectors = laplacian_eigenvalues(adjacency)
    cropped_eigenvectors = eigenvectors[:, 1:k]
    clusters = KMeans(n_clusters=k).fit_predict(cropped_eigenvectors)

    if k == 3:
        # plot eigenvalues 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cropped_eigenvectors[:, 0], cropped_eigenvectors[:, 1], cropped_eigenvectors[:, 2], c=clusters, cmap='viridis')
        plt.show()
    return clusters

def elbow_clusters(adjacency):
    sorted_eigenvalues, _ = laplacian_eigenvalues(adjacency)
    n_clusters, knee_value, x, y = elbow_cluster_count_from_eigvals(sorted_eigenvalues)
    print(f'Number of clusters: {n_clusters}, knee value: {knee_value}, first 10 eigvals: {sorted_eigenvalues[:10]}')

    #clusters = spectral_clustering(adjacency, n_clusters)
    return n_clusters

def eigengap_num_clusters(adjacency):
    eigenvalues, _ = laplacian_eigenvalues(adjacency)
    # crop out negative eigenvalues at the beginning
    if eigenvalues[0] < 0:
        # remove first X non-positive eigenvalues
        idx = np.where(eigenvalues > 0)[0][0]
        eigenvalues = eigenvalues[idx:]
    # else:
    #eigenvalues = eigenvalues[1:]
    # #gaps = np.diff(eigenvalues[1:])

    gap_index = plot_eigengaps(eigenvalues)
    return gap_index + 1

def clusters_from_adjacency(adjacency, cluster_method='eigengap', n_clusters=None):
    if n_clusters is None:
        if cluster_method == 'elbow':
            n_clusters = elbow_clusters(adjacency); print('using elbows')
        elif cluster_method == 'eigengap':
            n_clusters = eigengap_num_clusters(adjacency); print('using eigengap')
        else:
            raise ValueError('Unknown cluster method')
    print('num clusters', n_clusters)
    clusters = library_spectral_clustering(adjacency, n_clusters)
    return clusters

def get_clusters_from_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return np.array(data)

def calc_cluster_accuracy(corrs, sorted_indices, gt_clusters, n_clusters=None):
    corrs = corrs - np.diag(np.diag(corrs))
    abs_corr = np.abs(corrs)
    corr_clusters = clusters_from_adjacency(abs_corr, n_clusters=n_clusters)
    accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(gt_clusters, corr_clusters[sorted_indices])
    print(f'Clustering Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, FP rate: {fp_rate}, FP number: {fp}')


def save_adjacency_as_image(adjacency, fpath):
    cm = plt.get_cmap('viridis')
    colored = cm(adjacency)
    Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8)).save(fpath)
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(colored)
    plt.savefig(fpath)
    return colored

def save_clustered_adjacency_as_image(adjacency, clusters, fpath):
    # Number of clusters and a colormap avoiding red for intra-cluster connections
    n_clusters = len(np.unique(clusters))
    cluster_cmap = plt.cm.get_cmap('tab20b', n_clusters + 1)  # Skip red-like color
    
    # Prepare cluster color lookup, skipping the first color if it's red-like
    cluster_colors = cluster_cmap(np.arange(n_clusters) % cluster_cmap.N)[:,:3]
    
    # Create a boolean mask for within-cluster (True) and across-cluster (False) connections
    cluster_matrix = clusters[:, None] == clusters
    not_cluster_matrix = ~cluster_matrix
    
    # Initialize the image array using red for all connections, scaled by adjacency strength
    img = np.zeros(adjacency.shape + (3,))
    img[..., 0] = adjacency * not_cluster_matrix  # Red channel intensity for across-cluster connections
    
    # Apply cluster colors to within-cluster connections
    for i, color in enumerate(cluster_colors):
        mask = (clusters == i)[:, None] & (clusters == i)  # Create a mask for each cluster
        img[mask] = color  # Apply color to within-cluster connections
    
    # Normalize adjacency values for cross-cluster connections to maintain visibility
    img[not_cluster_matrix] *= img[not_cluster_matrix].max(axis=(0, 1), keepdims=True)
    
    # Save the image
    Image.fromarray((img * 255).astype(np.uint8)).save(fpath)
    plt.imshow(img)

def get_weather_station_clusters(column_names):
    # Extract the weather stations from the column names, such as:
    # AIR_TEMPERATURE_OH_Wooster_3_SSE', 'PRECIPITATION_OH_Wooster_3_SSE', 'SOLAR_RADIATION_OH_Wooster_3_SSE', ...

    variable_names = [
        'WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN', 'LONGITUDE', 'LATITUDE',
        'AIR_TEMPERATURE', 'PRECIPITATION', 'SOLAR_RADIATION', 'SR_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE', 'ST_FLAG',
        'RELATIVE_HUMIDITY', 'RH_FLAG', 'SOIL_MOISTURE_5', 'SOIL_TEMPERATURE_5', 'WETNESS', 'WET_FLAG', 'WIND_1_5', 'WIND_FLAG'
    ]

    weather_stations = []
    variables = []
    for column_name in column_names:
        found = False
        for variable_name in variable_names:
            if variable_name in column_name:
                station_name = column_name.split(variable_name + '_')[1]
                variables.append(variable_name)
                weather_stations.append(station_name)
                found = True
                break
        if not found:
            raise ValueError(f"Column name {column_name} not found")
    
    # replace station names with numbers
    num_stations = np.unique(np.array(weather_stations)).shape[0]

    # create cluster list
    clusters = np.zeros(len(column_names), dtype=int)

    # use np.unique to get a dict of station names and their indices
    stations = np.unique(np.array(weather_stations))
    station_idxs = {station: idx for idx, station in enumerate(stations)}

    # replace un-unique weather_stations with their index to get gt clusters

    weather_gt_clusters = np.sort(np.array([station_idxs[station] for station in weather_stations]))

    return weather_gt_clusters, num_stations

def max_perpendicular_point(x, y):
    start_point = np.array([x[0], y[0]])
    end_point = np.array([x[-1], y[-1]])
    line_vec = end_point - start_point
    print('line_vec', line_vec)
    line_unitvec = line_vec / np.linalg.norm(line_vec)
    print('line_unitvec', line_unitvec)

    # Compute the projection lengths of each point onto the line
    point_vecs = np.column_stack((x, y)) - start_point
    proj_lengths = np.dot(point_vecs, line_unitvec)

    # Find the perpendicular distances from the points to the line
    proj_points = np.outer(proj_lengths, line_unitvec) + start_point
    perp_distances = np.linalg.norm(point_vecs - proj_points, axis=1)

    # The knee is the point with the maximum perpendicular distance
    knee_index = np.argmax(perp_distances)
    knee_value = y[knee_index]
    max_distance = np.max(perp_distances)

    return knee_index, knee_value, max_distance

#%%
if __name__ == "__main__":
    pendulums = False
    us_weather = True
    if pendulums:
        dataset_name = 'pendulums'
        model_path = 'output/params/equinox/pendulums/20240613_02_14_46_0201-pretrain-best-e0-rebuttal_pendulum_mask_75_1e3_lr_16k_epoch_normal.eqx'
        own_adjacency = our_model_adjacency(model_path, dataset_name)

        clusters = clusters_from_adjacency(own_adjacency, n_clusters=3)
        pendulum_gt_clusters = get_clusters_from_sizes([4, 4, 4])
        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(pendulum_gt_clusters, clusters)
        print(f'Our model accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, FP rate: {fp_rate}')
        print('clusters', clusters)

        _, _, _, data = get_default_pendulum_data(lookback_size=512, forecast_size=96, batch_size=4096, t_stop=1_000, dt=0.025)
        print(data.shape)
        corrs = np.corrcoef(data.T)
        corrs = corrs - np.diag(np.diag(corrs))
        calc_cluster_accuracy(np.abs(corrs), np.arange(12), pendulum_gt_clusters, n_clusters=3)
        #sns.heatmap(corrs)


    #%%
    if us_weather:
        model_path = 'output/params/equinox/us_weather/20240606_17_22_45_5771-pretrain-best-e0-rebuttal_us_weather_mask_75_1e3_lr_200_epoch_normal.eqx'
        weather_adjacency = np.array(our_model_adjacency(model_path, dataset_name='us_weather'))

        wpath = 'output/plots/adjacency/us_weather/test_log1p.png'
        save_adjacency_as_image(np.log1p(weather_adjacency), wpath)
        #sns.heatmap(weather_adjacency)
        #%%
        import pandas as pd
        from config import US_WEATHER_DATA_PATH
        weather_df = pd.read_parquet(US_WEATHER_DATA_PATH)
        #%%
        xs = np.arange(weather_adjacency.flatten().shape[0])
        epsilon = 1e-6

        sorted_vals = np.sort(weather_adjacency.flatten())[::-1]
        ys = np.log1p(epsilon + sorted_vals)
        #ys = np.exp(epsilon + sorted_vals)
        #ys = np.sort(weather_adjacency.flatten())[::-1]
        plt.plot(xs, ys)
        #%%
        idx, val, max_distance = max_perpendicular_point(xs, ys)
        print(idx, val, max_distance, sorted_vals[idx])
        thresholded_weather_adj = weather_adjacency.copy()
        thresholded_weather_adj[weather_adjacency < sorted_vals[idx]] = 0
        (thresholded_weather_adj > 0).sum() / thresholded_weather_adj.ravel().shape[0]
        #%%
        weather_gt_clusters, num_stations = get_weather_station_clusters(weather_df.columns)
        #clusters = clusters_from_adjacency(weather_adjacency, cluster_method='elbow')
        clusters = clusters_from_adjacency(weather_adjacency, n_clusters=num_stations)
        #clusters = clusters_from_adjacency(thresholded_weather_adj, cluster_method='elbow')
        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(weather_gt_clusters, clusters)
        print(f'Our model Accuracy: {accuracy},\n Precision: {precision},\n Recall: {recall},\n F1: {f1},\n FP rate: {fp_rate},\n FP number: {fp}')
        #print('clusters', clusters)
        #%%
        numpy_df = weather_df.to_numpy()
        corrs = np.corrcoef(numpy_df.T)
        corrs = corrs - np.diag(np.diag(corrs))
        corrs = np.abs(corrs)
        #%%
        save_adjacency_as_image(corrs, 'output/plots/adjacency/us_weather/correlation_matrix.png')
        #%%
        corr_clusters = clusters_from_adjacency(corrs, cluster_method='elbow', n_clusters=num_stations)
        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(weather_gt_clusters, corr_clusters)
        print(f'Correlation matrix accuracy: {accuracy},\n Precision: {precision},\n Recall: {recall},\n F1: {f1},\n FP rate: {fp_rate},\n FP number: {fp}')
        #print('clusters', corr_clusters)