from torch.utils.data import Dataset
from utils.dataload_util import get_item
from utils.multi_slice_rows import get_sliding_windows_2d
import numpy as np

class ReconstructionWindowDataset(Dataset):
    def __init__(self, data, input_dim):
        self.data = data
        self.window = input_dim

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        return x, x, np.zeros_like(x)[:,:4] # use same data for input and output

    def __len__(self):
        return len(self.data) - self.window

class NonSlidingReconstructionWindowDataset(Dataset):
    def __init__(self, data, input_dim):
        self.data = data
        self.window = input_dim

    def __getitem__(self, index):
        # get the window with index
        x = self.data[index*self.window:(index+1)*self.window]
        return x, x, np.zeros_like(x)[:,:4]

    def __len__(self):
        return len(self.data) // self.window
    
class ForecastWindowDataset(Dataset):
    def __init__(self, data, data_time, input_dim, pred_len, set_name, random_timelag):
        self.input_len = input_dim
        self.pred_len = pred_len
        self.set_name = set_name

        input_dim = self.input_len + self.pred_len
        self.data_x = get_sliding_windows_2d(data.numpy(), input_dim)
        self.sensor_range = np.arange(self.data_x.shape[2])

        data_stamp = np.arange(len(data) * 2)[:, np.newaxis]
        self.data_stamp = get_sliding_windows_2d(data_stamp, input_dim)

        self.random_timelag = random_timelag

    def __getitem__(self, index):
        return get_item(self.input_len, self.data_x, self.data_stamp, self.random_timelag, index, self.sensor_range)

    def __len__(self):
        return self.data_x.shape[0]

class ForecastNoOverlapDataset(Dataset):
    def __init__(self, data, data_time, input_dim, pred_len, set_name, random_timelag, convert_to_windows=True):

        self.set_name = set_name
        self.data_x = data
        self.data_time = data_time
        self.input_len = input_dim
        self.pred_len = pred_len

        input_dim = self.input_len + self.pred_len

        if hasattr(data, 'numpy'):
            data = data.numpy()
        self.data_x = get_sliding_windows_2d(data, input_dim) if convert_to_windows else data
        self.sensor_range = np.arange(self.data_x.shape[2])

        data_stamp = np.arange(len(self.data_x) * 2)[:, np.newaxis]
        self.data_stamp = get_sliding_windows_2d(data_stamp, input_dim)

        self.random_timelag = random_timelag

    def __getitem__(self, index):
        if type(index) is tuple:
            index = (
                index[0] * self.pred_len,
                index[1],
                index[2],
            )
        else:
            index *= self.pred_len
        return get_item(self.input_len, self.data_x, self.data_stamp, self.random_timelag, index, self.sensor_range)

    def __len__(self):
        return self.data_x.shape[0] // self.pred_len
