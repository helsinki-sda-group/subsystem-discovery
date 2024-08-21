import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.functional import F


# MTGNN mock for creating adjacency
class GCModule(nn.Module):
    def __init__(self, num_nodes, emb_dim, lin1_in_features, lin1_out_features, lin2_in_features, lin2_out_features, alpha):
        super(GCModule, self).__init__()
        self.emb1 = nn.Embedding(num_nodes, emb_dim)
        self.emb2 = nn.Embedding(num_nodes, emb_dim)
        self.lin1 = nn.Linear(lin1_in_features, lin1_out_features)
        self.lin2 = nn.Linear(lin2_in_features, lin2_out_features)
        self.alpha = alpha

    def forward(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj

def df_to_corr_adj(df):
    corrs = np.corrcoef(np.array(df).T)
    corrs -= np.diag(np.diag(corrs))
    corrs = np.abs(corrs)
    return corrs


from config import US_WEATHER_DATA_PATH
def get_weather_df():
    return pd.read_parquet(US_WEATHER_DATA_PATH)

def create_adjacency_matrix_from_mi(df, num_vars):
    print('num vars', num_vars)
    matrix = np.zeros((num_vars, num_vars))
    for _, row in df.iterrows():
        index1 = int(row['Var1'])
        index2 = int(row['Var2'])
        mi = row['MI']
        matrix[index1, index2] = mi
        matrix[index2, index1] = mi  # Ensure the matrix is symmetric

    return matrix


def get_mtgnn_adjacency(state_dict_path, sorted_indices):
    state_dict = torch.load(state_dict_path, map_location='cpu')

    num_nodes = state_dict['gc.emb1.weight'].shape[0] 
    emb_dim = state_dict['gc.emb1.weight'].shape[1]
    lin1_in_features = state_dict['gc.lin1.weight'].shape[1]
    lin1_out_features = state_dict['gc.lin1.weight'].shape[0]
    lin2_in_features = state_dict['gc.lin2.weight'].shape[1]
    lin2_out_features = state_dict['gc.lin2.weight'].shape[0]
    alpha = 3  # Provided by you

    # Initialize the module
    gc = GCModule(num_nodes, emb_dim, lin1_in_features, lin1_out_features, lin2_in_features, lin2_out_features, alpha)

    param_names = [
        'gc.emb1.weight',
        'gc.emb2.weight',
        'gc.lin1.weight',
        'gc.lin1.bias',
        'gc.lin2.weight',
        'gc.lin2.bias'
    ]
    # Filter and adjust the keys in the loaded state_dict
    adjusted_state_dict = {k.replace('gc.', ''): v for k, v in state_dict.items() if k in param_names}
    gc.load_state_dict(adjusted_state_dict)

    idx = torch.arange(num_nodes)  # Example indices, adjust as necessary
    adj = gc(idx).detach()

    if adj.shape[0] == sorted_indices.shape[0]:
        print('sorting adjacency matrix')
        adj = adj[sorted_indices][:, sorted_indices]
    else:
        print('could not sort adjacency matrix, wrong shaped indices')


    adj = adj + adj.T 
    adj -= torch.diag(adj.diagonal())

    return adj