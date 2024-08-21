import numpy as np

def get_negative_sample(i, margin, N):
    valid_indices = np.concatenate([np.arange(0, i-margin), np.arange(i+margin, N)]) # arange (0, -5) = []
    if valid_indices.size == 0:
        raise Exception(f'No valid negative indices can be found for i {i}, margin {margin}, N {N}')
    return np.random.choice(valid_indices)


def get_item(input_len, data, data_stamp, random_timelag, index, precomp_sensors):
    if type(index) is tuple:
        index, label, overlap = index
    else:
        label = None
        overlap = 0

    if random_timelag:
        raise NotImplementedError('Not implemented')
    else:
        seq = data[index]
        seq_mark = data_stamp[index]
        gt_frac = np.zeros(seq.shape[1])

    seq_x = seq[:input_len]

    if label == 0:
        negative_index = get_negative_sample(index, margin=830, N=len(data))
        seq_y = data[negative_index, :input_len] # this part of data is supposed to be very far away, doesn't matter what chunk we take, hence :input_len
    else:
        seq_y_start = input_len - overlap
        seq_y = seq[seq_y_start:seq_y_start+input_len]
        #assert seq_y.shape[0] == input_len

    seq_x_mark  = seq_mark[:input_len]
    seq_y_mark = seq_mark[input_len:]

    return seq_x, seq_y, seq_x_mark, seq_y_mark, gt_frac

