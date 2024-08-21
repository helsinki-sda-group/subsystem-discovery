import torch
from torch.utils.data import DataLoader

from dataloaders.pendulum_dataloader import get_default_pendulum_data
from dataloaders.us_weather_dataset import get_us_weather_dataloader
from sklearn.utils import Bunch
from dataloaders.ltsf.ltsf_wrapper import load_ltsf_with_defaults
from dataloaders.windowing import ForecastNoOverlapDataset, ForecastWindowDataset

def get_dataloader(**kwargs):
    if kwargs['dataset_name'] == 'us_weather':
        args = Bunch(**kwargs)
        train_loader, val_loader, test_loader, pred_loader, labels, train_data = get_us_weather_dataloader(args, wrap_to_future_predict_dataloader)
        n_sensors = train_data.shape[2]
        return train_data, train_loader, val_loader, test_loader, pred_loader, n_sensors
    if kwargs['dataset_name'] == 'pendulums':
        train_loader, val_loader, test_loader, full_data = get_default_pendulum_data(lookback_size=kwargs['input_dim'], forecast_size=kwargs['pred_len'], batch_size=kwargs['batch_size'], t_stop=kwargs['t_stop'], dt=kwargs['dt'])
        n_sensors = full_data.shape[-1]
        print('pendulum data shape', full_data.shape)
        return full_data, train_loader, val_loader, test_loader, None, n_sensors
    train_data, val_data, test_data, train_loader_ltsf, val_loader_ltsf, test_loader_ltsf, pred_loader_ltsf = load_ltsf_with_defaults(
        **kwargs
    )

    index, label, overlap = 0, 0, kwargs['overlap_len']
    n_sensors = train_data[(index, label, overlap)][0].shape[1]
    return train_data, train_loader_ltsf, val_loader_ltsf, test_loader_ltsf, pred_loader_ltsf, n_sensors

def wrap_to_future_predict_dataloader(x, y, args):
    val_len = 1.0 * args.test_len
    val_num = int(x.shape[0] * val_len)

    train_num = int(x.shape[0] * (1 - args.test_len - val_len))
    test_num = x.shape[0] - train_num - val_num

    target_train = torch.from_numpy(y[:train_num]).float()
    target_val = torch.from_numpy(y[train_num:train_num+val_num]).float()
    target_test = torch.from_numpy(y[train_num+val_num:train_num+val_num+test_num]).float()
    target_pred = target_test

    dataset_class = ForecastNoOverlapDataset if args.no_overlap_override else ForecastWindowDataset

    return get_pytorch_dataloader(target_train, target_val, target_test, target_pred, dataset_class, args)

def get_pytorch_dataloader(target_train, target_val, target_test, target_pred, dataset_class, args):
    dataset = dataset_class(target_train, data_time=None, input_dim=args.input_dim, pred_len=args.pred_len, set_name='train', random_timelag=args.random_timelag)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(not args.no_overlap_override), num_workers=0, drop_last=True)

    val_dataset = dataset_class(target_val, data_time=None, input_dim=args.input_dim, pred_len=args.pred_len, set_name='val', random_timelag=args.random_timelag)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // 8, shuffle=False, num_workers=0, drop_last=True)

    test_dataset = dataset_class(target_test, data_time=None, input_dim=args.input_dim, pred_len=args.pred_len, set_name='test', random_timelag=args.random_timelag)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size // 8, shuffle=False, num_workers=0)

    pred_dataset = ForecastNoOverlapDataset(target_pred, data_time=None, input_dim=args.input_dim, pred_len=args.pred_len, set_name='pred', random_timelag=args.random_timelag)
    pred_loader = DataLoader(pred_dataset, batch_size=args.batch_size // 8, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, pred_loader


def setup_dataloader(config):
    train_data, train_loader_ltsf, val_loader_ltsf, test_loader_ltsf, pred_loader_ltsf, n_sensors = get_dataloader(
        **config)

    return {
        **config,
        "n_sensors": n_sensors,
        "train_loader": train_loader_ltsf,
        "val_loader": val_loader_ltsf,
        "test_loader": test_loader_ltsf,
        "pred_loader": pred_loader_ltsf,
    }
