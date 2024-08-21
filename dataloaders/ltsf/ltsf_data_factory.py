from dataloaders.ltsf.ltsf_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

from dataloaders.windowing import ForecastNoOverlapDataset

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, dataloader=DataLoader):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        freq = args.freq

    data_set = Data(
        random_timelag=args.random_timelag,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    if flag == 'pred' or args.no_overlap_override:
        shuffle_flag = False
        drop_last = False
        data_set = ForecastNoOverlapDataset(
            data_set.data_x, data_set.data_stamp, args.seq_len, args.pred_len, data_set.set_name, 
            random_timelag=args.random_timelag,
            convert_to_windows=True if (len(data_set.data_x.shape) < 3) else False)

    print(flag, len(data_set))
    data_loader = dataloader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last) # Drop always to keep compatibility with Jax
    return data_set, data_loader
