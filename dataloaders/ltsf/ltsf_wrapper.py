# Bridge between our dataloaders and the ones in "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023) https://github.com/cure-lab/LTSF-Linear

from sklearn.utils import Bunch
from torch.utils.data import DataLoader

from config import LTSF_DATA_ROOT_PATH, NUM_WORKERS
from dataloaders.ltsf.ltsf_data_factory import data_provider

dataset_mapping = {
    'electricity':          ['electricity.csv', 'custom'],
    'exchange':             ['exchange_rate.csv', 'custom'],
    'national_illness':     ['national_illness.csv', 'custom'],
    'traffic':              ['traffic.csv', 'custom'],
    'weather':              ['weather.csv', 'custom'],
    'ETTh1':                ['ETTh1.csv', 'ETTh1'],
    'ETTh2':                ['ETTh2.csv', 'ETTh2'],
    'ETTm1':                ['ETTm1.csv', 'ETTm1'],
    'ETTm2':                ['ETTm2.csv', 'ETTm2'],
}

# args is our commandline params, mapping them to ltsf
def load_ltsf_dataset(args):

    embed = 'timeF' 
    label_len = 0 
    ltsf_args = {
        'random_lag': args.random_timelag,
        'root_path': LTSF_DATA_ROOT_PATH,
        'data_path': dataset_mapping[args.dataset][0],
        'data': dataset_mapping[args.dataset][1],
        'features': 'S' if args.univariate else 'M', # S = univariate, M=multi
        'embed': embed, 
        'target': 'OT', 
        'freq': 'h', 
        'seq_len': args.input_dim,
        'label_len': label_len, # wut
        'pred_len': args.pred_len,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    ltsf_args = Bunch(**ltsf_args)

    # Flag can be 'train', 'val', 'test', 'pred'
    train_data_set, train_data_loader = data_provider(ltsf_args, 'train')
    val_data_set, val_data_loader = data_provider(ltsf_args, 'val')
    test_data_set, test_data_loader = data_provider(ltsf_args, 'test')
    pred_data_set, pred_data_loader = data_provider(ltsf_args, 'pred')

    feature_names = None # TODO
    #jreturn train_loader, test_loader, feature_names
    return train_data_loader, test_data_loader, val_data_loader, pred_data_loader, feature_names, train_data_set.data_x
    
def load_ltsf_with_defaults(dataset_name='ETTm1', univariate=False, input_dim=336, dataloader=DataLoader, random_timelag=False, pos_enc_on=False, **kwargs):

    ltsf_args = {
        'random_timelag': random_timelag,
        'pos_enc_on': pos_enc_on,
        'root_path': LTSF_DATA_ROOT_PATH,
        'data_path': dataset_mapping[dataset_name][0],
        'data': dataset_mapping[dataset_name][1],
        'features': 'S' if univariate else 'M', # S = univariate, M=multi
        'embed': 'asdfas', 
        'target': 'OT', 
        'freq': 'h', 
        'seq_len': input_dim,
        'label_len': 0, 
        'num_workers': NUM_WORKERS,
        **kwargs
    }
    ltsf_args = Bunch(**ltsf_args)

    train_data, train_loader_ltsf = data_provider(ltsf_args, 'train', dataloader=dataloader)
    test_data, test_loader_ltsf = data_provider(ltsf_args, 'test', dataloader=dataloader)
    val_data, val_loader_ltsf = data_provider(ltsf_args, 'val', dataloader=dataloader)
    _, pred_loader_ltsf = data_provider(ltsf_args, 'pred', dataloader=dataloader)

    return train_data, val_data, test_data, train_loader_ltsf, val_loader_ltsf, test_loader_ltsf, pred_loader_ltsf