
#%%
import pandas as pd
import numpy as np
from evaluation.create_adjacencies import get_mtgnn_adjacency, get_weather_df, df_to_corr_adj, create_adjacency_matrix_from_mi
from evaluation.clustering_funcs import our_model_adjacency
#%%

########################################
# Our model
########################################

def get_ours():
    our_paths = [
        'evaluation/proposed/20240613_02_14_46_0201-pretrain-best-e0-rebuttal_pendulum_mask_75_1e3_lr_16k_epoch_normal.eqx',
        'evaluation/proposed/20240606_17_22_45_5771-m200-mf0-f0-da0-pretrain-last-rebuttal_us_weather_mask_75_1e3_lr_200_epoch_normal.eqx'
    ]

    our_adjs = [our_model_adjacency(path) for path in our_paths]
    return our_adjs

########################################
# MTGNN
########################################
def get_mtgnns():
    mtgnn_paths = [
       'evaluation/mtgnn/pendulum.pth',
       'evaluation/mtgnn/mtgnn_us_weather.pth'
    ]
    return [get_mtgnn_adjacency(path, np.zeros(1)) for path in mtgnn_paths]


########################################
# Correlation matrix
########################################

def get_corrs():
    print('get corrs, this will take a while')
    corrs = [
        df_to_corr_adj(pd.read_csv('data/pendulum_data.txt')), # full data not training set!
        df_to_corr_adj(get_weather_df()),
    ]
    print('corrs done')

    return corrs

def get_mutuals():
    mutual_files = [
        'evaluation/mutual_information/mi_results_combined_12_vars.csv',
        'evaluation/mutual_information/mi_results_combined_987_vars.csv'
    ]

    adjacencies = []
    for file in mutual_files:
        df = pd.read_csv(file)
        num_vars = int(file.split('_')[-2])
        adj = create_adjacency_matrix_from_mi(df, num_vars)
        adjacencies.append(adj)

    return adjacencies