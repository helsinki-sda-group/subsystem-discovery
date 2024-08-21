#%%
import torch
import matplotlib.pyplot as plt

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
state_dict_path = 'save/pendulum-long/exp1_0_epoch_200.pth'
state_dict = torch.load(state_dict_path, map_location='cpu')

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

        # mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))
        # s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        # mask.scatter_(1,t1,s1.fill_(1))
        # adj = adj*mask

        return adj



# Assuming shapes from the state_dict you've printed
num_nodes = state_dict['gc.emb1.weight'].shape[0] 
emb_dim = state_dict['gc.emb1.weight'].shape[1]
lin1_in_features = state_dict['gc.lin1.weight'].shape[1]
lin1_out_features = state_dict['gc.lin1.weight'].shape[0]
lin2_in_features = state_dict['gc.lin2.weight'].shape[1]
lin2_out_features = state_dict['gc.lin2.weight'].shape[0]
alpha = 3  # Provided by you

# Initialize the module
gc = GCModule(num_nodes, emb_dim, lin1_in_features, lin1_out_features, lin2_in_features, lin2_out_features, alpha)

#%%
# Filter and adjust the keys in the loaded state_dict
adjusted_state_dict = {k.replace('gc.', ''): v for k, v in state_dict.items() if k in param_names}
gc.load_state_dict(adjusted_state_dict)

# %%
def plot_adj(adj):
    # Symmetrize the adjacency matrix
    adj = adj + adj.T - torch.diag(adj.diagonal())
    
    plt.imshow(adj, cmap='viridis')

    if adj.shape[0] == 12:
        # Custom tick labels for three double pendulums
        tick_labels = ['P1_x1', 'P1_y1', 'P1_x2', 'P1_y2', 'P2_x1', 'P2_y1', 'P2_x2', 'P2_y2', 'P3_x1', 'P3_y1', 'P3_x2', 'P3_y2']
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=90)
        plt.yticks(range(len(tick_labels)), tick_labels)

    plt.colorbar()
    plt.show()

idx = torch.arange(num_nodes)  # Example indices, adjust as necessary
adj = gc(idx).detach()
plot_adj(adj)
# %%
