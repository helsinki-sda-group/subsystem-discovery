#%%
import numbers
import numpy as np
from numpy.lib.stride_tricks import as_strided
#%%    
# view_as_windows Copied from scikit:
# https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py#L97C1-L97C1
def view_as_windows(arr_in, window_shape, step=1):
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out

def get_sliding_windows_2d(X, input_dim, step=1):
    return view_as_windows(X, (input_dim, X.shape[1]), step=step)[:, 0]

def get_windows(precomputed_sliding_window, precomp_sensors, time_indices, input_dim, num_windows, random_seed=None):
    result = precomputed_sliding_window[time_indices, :, precomp_sensors]
    return result.copy().T

# Example usage
if __name__ == '__main__':
    # Get sliding windows
    num_sensors = 10
    num_timesteps = 1000
    input_dim = 8
    #X = np.random.rand(num_timesteps, num_sensors)
    X = np.arange(num_timesteps) # [1,2, 3,...]
    X = np.tile(X, (num_sensors, 1)).T  #[[1,1,1],[2,2,2],...]
    X.shape
    #%%
    w = view_as_windows(X, (input_dim, num_sensors))[:, 0]
    w.shape
    #%%
    time_indices = np.random.randint(0,w.shape[0], num_sensors)
    picked_sensors = np.arange(num_sensors) # pick all sensors
    result = w[time_indices, :, picked_sensors]
    print(result.shape)
    #%%
    print(result)


    ### And same with the predone functions
    precomputed_sliding_window = get_sliding_windows_2d(X, input_dim)
    precomp_sensors = np.arange(num_sensors)
    # pick one random time index per sensor
    time_indices = np.random.randint(0,precomputed_sliding_window.shape[0], num_sensors) 
    result = get_windows(precomputed_sliding_window, precomp_sensors, time_indices, input_dim, num_sensors)
    print(result.shape)
    print(result)