#%%
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from evaluation.clustering_funcs import get_clusters_from_sizes, get_weather_station_clusters, clusters_from_adjacency, save_clustered_adjacency_as_image, max_perpendicular_point
from evaluation.clustering_accuracy_metrics import clustering_accuracy, get_clusters_from_sizes
from evaluation.evaluation_adjacencies import get_mtgnns, get_ours, get_mutuals
from evaluation.create_adjacencies import df_to_corr_adj, get_weather_df


from plot_weather_adjacencies import get_fiedler_indices
#%%
def find_highest_epoch_number(save_path, expid):
    highest_epoch = -1
    pattern = f"{save_path}exp{expid}_*_epoch_*.pth"
    for filename in glob(pattern):
        parts = filename.split('_')
        epoch_num = int(parts[-1].split('.')[0])
        highest_epoch = max(highest_epoch, epoch_num)
    if highest_epoch == -1:
        raise ValueError(f"No saved models found in {save_path}")
    return highest_epoch

import matplotlib.colors as colors
def plot_adj(adj, name):
    print('plotting', name)
    #plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(15, 15))
    gamma = 0.5
    #plt.imshow(adj, cmap='viridis')
    ax.imshow(adj, norm=colors.PowerNorm(gamma=gamma), cmap='viridis')
    #ax.imshow(adj, cmap='viridis')

    ax.set_xticks([])
    ax.set_yticks([])

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    if adj.shape[0] == 12:
        # Custom tick labels for three double pendulums
        tick_labels = ['P1_x1', 'P1_y1', 'P1_x2', 'P1_y2', 'P2_x1', 'P2_y1', 'P2_x2', 'P2_y2', 'P3_x1', 'P3_y1', 'P3_x2', 'P3_y2']
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=90)
        plt.yticks(range(len(tick_labels)), tick_labels)

    #plt.savefig(f'evaluation/adjacencies/{name}.pdf', bbox_inches='tight')
    plt.savefig(f'evaluation/adjacencies/{name}.png', bbox_inches='tight')

def get_weather_clusters():
    df = get_weather_df()
    corrs = df_to_corr_adj(df)
    gt_clusters, num_stations = get_weather_station_clusters(df.columns)
    return gt_clusters, num_stations, corrs

def knee_threshold(adj):
    ys = np.sort(adj.flatten())[::-1]
    x, y = np.arange(len(ys)), ys
    knee_idx, knee_value, dist = max_perpendicular_point(x, y)

    fig = plt.figure()
    plt.plot(x,y)
    plt.plot(knee_idx, knee_value, 'ro')
    #plt.show()

    new_adj = adj.copy()
    new_adj[new_adj < knee_value] = 0
    return new_adj

only_plot = False
pred_num_clusters = False
cluster_method = 'eigengap'
from config import US_WEATHER_DATA_PATH
if __name__ == '__main__':
    weather_gts, num_stations, weather_corrs = get_weather_clusters()
    stations_path = './data/stations.tsv'
    stations_data = pd.read_csv(stations_path, delimiter='\t')
    df = pd.read_parquet(US_WEATHER_DATA_PATH)
    indices = get_fiedler_indices(stations_data, df)[0]
    pendulum_gts = get_clusters_from_sizes([4, 4, 4])
    # MTGNN

    print("MTGNN results")
    mtgnn_models = get_mtgnns()
    dicts = [
        ['mtgnn-pendulum', mtgnn_models[0], pendulum_gts, 3],
        ['mtgnn-us_weather', mtgnn_models[1], weather_gts, num_stations]
    ]
    plt.clf()
    for name, adj, gts, num_clusters in dicts:
        adj = np.array(adj)
        plot_adj(adj, name)
        if only_plot:
            continue

        if pred_num_clusters:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=None)
            continue
        else:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=num_clusters)
        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(gts, clusters)

        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, False positive rate: {fp_rate}, False positives: {fp}')

        # print in format      & MTGNN     & 0.7555 & 0.143 & 0.144 & 0.143 & 0.143 & 242613 \\ \cline{2-8} 
        print(f"& MTGNN & {accuracy:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {fp_rate:.3f} & {fp} \\\\", '\\cline{2-8}')


        filepath = f'evaluation/adjacencies/{name}-clustered.png'
        save_clustered_adjacency_as_image(adj, clusters, filepath)
        print('filepath', filepath)

    # %%
    mutual_adjacencies = get_mutuals()

    for adj, gts, num_clusters in zip(mutual_adjacencies, [pendulum_gts, weather_gts], [3, num_stations]):
        plot_adj(adj, f'mutual_information_{num_clusters}')
        if only_plot:
            continue
        if pred_num_clusters:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=None)
            continue
        else:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=num_clusters)

        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(gts, clusters)
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, False positive rate: {fp_rate}, False positives: {fp}')
        print(f"& Kraskov MI & {accuracy:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {fp_rate:.3f} & {fp} \\\\", '\\cline{2-8}')
        save_clustered_adjacency_as_image(adj, clusters, f'evaluation/adjacencies/mutual_information_{num_stations}_clustered.png')


    print('proposed model results')
    proposed = get_ours()
    own_models = [
        [proposed[0], 3, 'pendulums', pendulum_gts],
        [proposed[1], num_stations, 'us_weather', weather_gts],
    ]

    #%%
    plt.clf()
    for (adj, num_clusters, dataset_name, gts) in own_models:
        adj = np.array(adj)
        fig = plt.figure(figsize=(20, 20))
        cm = plt.get_cmap('viridis')
        colored = cm(adj)
        #Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8))
        plt.imshow(colored)
        plt.savefig(f'evaluation/adjacencies/proposed_{dataset_name}.png')
        #plt.show()

        if only_plot:
            continue

        if adj.shape[0] == 987:
            adj = adj[indices][:, indices]
            gts = gts[indices]

        if pred_num_clusters:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=None)
            continue
        else:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=num_clusters)

        accuracy, precision, recall, f1, fp_rate, fp = clustering_accuracy(gts, clusters)
        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, False positive rate: {fp_rate}, False positives: {fp}')
        print(f"& Ours & {accuracy:.3f} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {fp_rate:.3f} & {fp} \\\\", '\\cline{2-8}')

        # filename = model_path.split('/')[-1].split('.')[0]
        save_clustered_adjacency_as_image(adj, clusters, f'evaluation/adjacencies/proposed_{dataset_name}_clustered.png')
    # %%

    print("Correlation stuff")
    pendulum_corrs = np.corrcoef(np.array(pd.read_csv('data/pendulum_data.txt')).T)
    np.set_printoptions(suppress=True)
    corrs = {
        'pendulums': pendulum_corrs,
        'us_weather': weather_corrs
    }
    for gts, num_clusters, dataset_name, _ in own_models:
        adj = corrs[dataset_name]
        adj -= np.diag(np.diag(adj))
        adj += np.random.normal(0, 0.01, adj.shape)
        adj = np.abs(adj)
        plot_adj(adj, 'corr_' + dataset_name)
        if only_plot:
            continue

        if pred_num_clusters:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=None)
            continue
        else:
            clusters = clusters_from_adjacency(adj, cluster_method=cluster_method, n_clusters=num_clusters)