import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from tqdm import tqdm


#device = 'cuda'
device = 'cpu'

def create_sequences(data, seq_in, seq_out):
    x, y = [], []
    for i in tqdm(range(len(data) - seq_in - seq_out + 1)):
        x.append(data[i:(i + seq_in)])
        y.append(data[(i + seq_in):(i + seq_in + seq_out)])
    return np.array(x), np.array(y)

def process_and_save_data(source, seq_in, seq_out, train_ratio=0.8, val_ratio=0.10, dataset_dir='data'):
    # Load data
    print('load data')
    if isinstance(source, str):
        if source.endswith('.parquet'):
            data = pd.read_parquet(source)
        elif source.endswith('.csv') or source.endswith('.txt'):
            data = pd.read_csv(source)
        else:
            raise ValueError("Unsupported file format")
    else:
        data = source
    
    print('data loaded, add features')
    data = data.values.reshape(-1, data.shape[1], 1)  # Add feature dimension

    # Create sequences
    print('create seq')
    x, y = create_sequences(data, seq_in, seq_out)

    print('train val test split')
    # Split into training, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=1-train_ratio, random_state=42)
    test_ratio = val_ratio / (1 - train_ratio)  # Adjust test_ratio
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=test_ratio, random_state=42)

    print('save result to files')
    # Save to .npz files
    np.savez_compressed(os.path.join(dataset_dir, 'test.npz'), x=x_test, y=y_test)
    np.savez_compressed(os.path.join(dataset_dir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(dataset_dir, 'train.npz'), x=x_train, y=y_train)

    print("Datasets saved in:", dataset_dir)

def data_preprocess(data_path, args):
    if 'pendulum' in data_path:
        store_path = f'./data/{"pendulum" if "pendulum" in data_path else "powerplant"}_in_{args.seq_in_len}_out_{args.seq_out_len}/'
        if os.path.exists(store_path):
            print('Data already preprocessed')
            return store_path
        os.makedirs(store_path)
        process_and_save_data(data_path, seq_in=args.seq_in_len, seq_out=args.seq_out_len, dataset_dir=store_path)
    else:
        raise ValueError("Unsupported dataset")
    
    return store_path

def get_data(source):
    print('load data')
    if isinstance(source, str):
        if source.endswith('.parquet'):
            data = pd.read_parquet(source)
        elif source.endswith('.csv') or source.endswith('.txt'):
            data = pd.read_csv(source)
        else:
            raise ValueError("Unsupported file format")
    else:
        data = source

    return data

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
    
    def __len__(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, data, two_col):
        self.per_variable = True

        if self.per_variable:
            if two_col:
                xmean = torch.tensor(data).mean(axis=0).reshape(1, 1, -1, 1) # [1, nodes], want into [1, 1, nodes, 1]
                xstd = torch.tensor(data).std(axis=0).reshape(1,1,-1,1)
            else:
                assert data.ndim == 4 # [full seq, seq len, nodes, features
                xmean = torch.tensor(data).mean(axis=(0, 1)).reshape(1, 1, -1, 1) # [1, nodes], want into [1, 1, nodes, 1]0
                xstd = torch.tensor(data).std(axis=(0, 1)).reshape(1, 1, -1, 1) # [1, nodes], want into [1, 1, nodes, 1]0
        else:
            xmean = torch.tensor(data).mean() # [1, nodes], want into [1, 1, nodes, 1]
            xstd = torch.tensor(data).std()

        self.mean = xmean.to(device)
        self.std = xstd.to(device)
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data, idx): # data eg [64, 1, 12, 12]), [batch, feat, nodes, seq]
        if self.per_variable:
            return (data * self.std[:, :, idx]) + self.mean[:, :, idx]
        else:
            return (data * self.std) + self.mean

class DataLoaderCustom(object):
    def __init__(self, data, batch_size, seq_in, seq_out, scaler, pad_with_last_sample=False):
        """
        :param data: Input data as a pandas DataFrame converted to a PyTorch tensor.
        :param batch_size: Number of samples per batch.
        :param seq_in: Input sequence length.
        :param seq_out: Output sequence length.
        :param scaler: Scaler object which is now a PyTorch module to normalize data.
        :param pad_with_last_sample: Pad with the last sample to make number of samples divisible by batch_size.
        """
        self.data = torch.tensor(data).float().unsqueeze(-1)  # Ensure data has shape [N, features, 1] if not already
        self.batch_size = batch_size
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.scaler = scaler
        
        self.total_size = len(self.data) - self.seq_in - self.seq_out + 1
        if pad_with_last_sample:
            num_padding = (batch_size - (self.total_size % batch_size)) % batch_size
            if num_padding > 0:
                padding = self.data[-1:].repeat(num_padding, 1, 1)
                self.data = torch.cat([self.data, padding], dim=0)
                self.total_size += num_padding

        self.permutation = torch.randperm(self.total_size)

    def shuffle(self):
        # Shuffling the indices from which sequences start
        self.permutation = torch.randperm(self.total_size)
    
    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.total_size:
                start_indices = self.permutation[self.current_ind:self.current_ind + self.batch_size]
                x_indices = start_indices[:, None] + torch.arange(self.seq_in)
                y_indices = start_indices[:, None] + self.seq_in + torch.arange(self.seq_out)

                x_batch = self.data[x_indices]
                y_batch = self.data[y_indices]

                # Applying the scaler transform here
                x_batch_normalized = self.scaler.transform(x_batch.to(device))

                yield (x_batch_normalized, y_batch)
                self.current_ind += self.batch_size
        return _wrapper()

def load_dataframe(dataframe, batch_size, seq_in, seq_out, valid_batch_size=None, test_batch_size=None):
    if valid_batch_size is None:
        valid_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size

    n = len(dataframe)
    #n = 25_000

    train_df = dataframe.iloc[:int(0.8 * n)]

    scaler = StandardScaler(np.array(train_df), two_col=True)
    #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    valid_df = dataframe.iloc[int(0.8 * n):int(0.9 * n)]

    end_idx = int(0.9 * n) + valid_df.shape[0]
    test_df = dataframe.iloc[int(0.9 * n):end_idx]

    # Create data arrays from DataFrames
    x_train = train_df.values
    x_val = valid_df.values
    x_test = test_df.values

    # Create data loaders
    train_loader = DataLoaderCustom(x_train, batch_size, seq_in, seq_out, scaler)
    val_loader = DataLoaderCustom(x_val, valid_batch_size, seq_in, seq_out, scaler)
    test_loader = DataLoaderCustom(x_test, test_batch_size, seq_in, seq_out, scaler)

    data = {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
    }

    return data

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# data_loaders = load_dataframe(df, batch_size=64, seq_in=512, seq_out=10)




def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(data['x_train'], two_col=False) # (27983, 12, 12, 1) for pendulums, (23974, 12, 207, 2) for METR-LA
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0:1] = scaler.transform(torch.tensor(data['x_' + category][..., 0:1]).to(device)).cpu()

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))



            