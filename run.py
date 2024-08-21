#%%
import argparse
import datetime
import os
import jax
import optax
import equinox as eqx
from sklearn.utils import Bunch

from model.model import Maskformer, default_mask_model_settings
from training.train_fun import train_plot_save
from dataloaders.dataloader import setup_dataloader
from training.train_fun import evaluate, count_params
from utils.misc import log_params, get_tensorboard_writer


#%%
def get_params():
    n_devices = len(jax.devices())
    if n_devices > 1:
        print('Using', n_devices, 'GPUs')
        pass
        #evaluate = evaluate_multigpu

    parser = argparse.ArgumentParser(description='Arguments for the script.')
    parser.add_argument('--seed', default=1337, type=int, help='Seed value')
    parser.add_argument('--script_name', default='default', type=str, help='Seed value')

    parser.add_argument('--batch_size', default=32 * n_devices, type=int, help='Seed value')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='Seed value')
    parser.add_argument('--learning_rate_schedule_steps', default=0, type=float, help='Seed value')
    parser.add_argument('--learning_rate_schedule_multiplier', default=10, type=float, help='Seed value')

    parser.add_argument('--input_dim', default=256, type=int, help='Seed value')
    parser.add_argument('--pred_len', default=64, type=int, help='Seed value')

    parser.add_argument('--dataset_name', default='us_weather', type=str, help='Seed value')

    parser.add_argument('--t_stop', default=1000, type=int, help='Seed value')
    parser.add_argument('--dt', default=0.025, type=float, help='Seed value')

    parser.add_argument('--patch_len', default=16, type=int, help='Seed value')
    parser.add_argument('--n_layers', default=4, type=int, help='Seed value')
    parser.add_argument('--n_decoder_layers', default=1, type=int, help='Seed value')
    parser.add_argument('--n_heads', default=4, type=int, help='Seed value')
    parser.add_argument('--forecast_hidden_dim', default=2048, type=int, help='Seed value')

    parser.add_argument('--rank_div', default=16, type=int, help='Seed value')

    parser.add_argument('--revin', default=False, action='store_true', help='Normalization flag')
    parser.add_argument('--instance_norm', default=False, action='store_true', help='Normalization flag')

    # Powerplant settings
    parser.add_argument('--power_plant_max_data_size', default=1_800_000, type=int, help='Seed value')
    parser.add_argument('--power_plant_max_cols', default=2_000, type=int, help='Seed value')
    parser.add_argument('--downsampler', default='30s', type=str, help='Seed value')
    parser.add_argument('--power_plant_n_engines', default=7, type=int, help='Seed value')
    parser.add_argument('--rolling_mean', default=1, type=int, help='Seed value')
    parser.add_argument('--normalize', default=False, action='store_true', help='Normalization flag')
    parser.add_argument('--test_len', default=0.10, type=float, help='Seed value')
    parser.add_argument('--cache_df', default=False, action='store_true', help='Normalization flag')

    # Unused stuff
    parser.add_argument('--no_overlap_override', default=False, action='store_true', help='Normalization flag')
    parser.add_argument('--random_timelag', default=False, action='store_true', help='Normalization flag')
    parser.add_argument('--overlap_len', default=0, type=int, help='Seed value')

    parser.add_argument('--mask_p', default=0.75, type=float, help='Seed value')
    parser.add_argument('--l1_regularize', default=0.0, type=float, help='Seed value')
    parser.add_argument('--eigval_reg', default=0.0, type=float, help='Seed value')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Seed value')

    parser.add_argument('--epochs', default=2, type=int, help='Seed value')
    parser.add_argument('--masked_forecast_epochs', default=0, type=int, help='Seed value')
    parser.add_argument('--forecast_epochs', default=2, type=int, help='Seed value')
    parser.add_argument('--n_forecast_layers', default=0, type=int, help='Seed value')

    parser.add_argument('--disable_adjacency', default=False, action='store_true', help='Model is completely univariate if true')

    parser.add_argument('--model_pretrain_path', default='', type=str, help='Seed value')

    args = parser.parse_args()
    return args


def save_params(model_name, model, model_settings, timestamp, dataset_name):
    # if directory missing, create it
    dirname = f'output/params/equinox/{dataset_name}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    path = f'{dirname}/{timestamp}-{model_name}.eqx'
    eqx.tree_serialise_leaves(path, model)

    if 'key' in model_settings:
        del model_settings['key']
    import json
    with open(f'{path}-settings.json', 'w') as f:
        json.dump(model_settings, f)
    
    print(f'saved model to {path}')
    return path

def run_train():
    # clear commandline arguments for jupyter
    #sys.argv = ['']
    # Disable JIT globally
    # from jax.config import config as jaxconfig
    #jaxconfig.update('jax_disable_jit', True)
    #args.batch_size = 256 # for one engine

    args = get_params()

    config = Bunch(**args.__dict__)

    config = Bunch(**setup_dataloader(config))
    n_sensors = config.n_sensors
    print(config)

    ADJ_OUTPUT_PATH=f'output/plots/adjacency/{config.dataset_name}/{n_sensors}_sensors/'
    config.adj_output_path = ADJ_OUTPUT_PATH
    if not os.path.exists(ADJ_OUTPUT_PATH):
        os.makedirs(ADJ_OUTPUT_PATH)

    input_dim = config.input_dim
    pred_len = config.pred_len
    patch_len = 16
    config.patch_len = patch_len
    learning_rate = config.learning_rate

    if config.learning_rate_schedule_steps > 0:
        schedule_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=learning_rate * config.learning_rate_schedule_multiplier,
            transition_steps=config.learning_rate_schedule_steps
        )
        config.learning_rate = schedule_fn

    if config.weight_decay > 0:
        optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optax.adam(learning_rate=config.learning_rate)


    train_loader, val_loader, test_loader = config.train_loader, config.val_loader, config.test_loader

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    model_settings = default_mask_model_settings(config.dataset_name, subkey, config, n_sensors)

    # Initialize the model
    model = Maskformer(**model_settings)
    from model.model import get_loss_fn # nasty dependency :(
    loss_fn = get_loss_fn(config)

    # Note that this will not load any forecaster head MLP weights even if they are in the path, as the model definition here is without that part!
    # so its mostly meant for pretraining the encoder part MAE style.
    if args.model_pretrain_path != '':
        print('------------------ loading pretrained model ------------------')
        print(args.model_pretrain_path)
        model = eqx.tree_deserialise_leaves(args.model_pretrain_path, model)
        print('------------------ loaded pretrained model ------------------')

    model_params, _ = eqx.partition(model, eqx.is_array)
    count_params(model)

    writer = get_tensorboard_writer('logs', name=f'{args.script_name}')
    key = jax.random.split(key)[0]
    #print_every = len(train_loader) - 1
    #print_every = len(train_loader) // 2
    print_every = len(train_loader) // 2 + 1

    # timestamp
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S") # share timestamp for all model saves
    # version with milliseconds:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")[:-2]
    print('timestamp', timestamp)
    config.timestamp = timestamp

    if config.epochs > 0:
        print('------------------ pretraining ------------------')
        epochs = config.epochs
        save_params_fn = lambda model, epoch, state: save_params(f'pretrain-{state}-e{epoch}-{args.script_name}', model, model_settings, timestamp, config.dataset_name)
        model, model_params, model_static = train_plot_save('pretrain', config, model_settings, train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, key=key, epochs=epochs, print_every=print_every)
        print('------------------ pretraining done ------------------')
        #model = train_and_evaluate(train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, key=key, epochs=epochs, print_every=print_every)

    if config.masked_forecast_epochs > 0:
        print('------------------ masked_forecast ------------------')
        epochs = config.masked_forecast_epochs
        optimizer = optax.adam(learning_rate=1e-3)
        key = jax.random.split(key)[0]

        # always mask end but also mask random patches
        model_settings['key'] = key
        model_settings['mask_end'] = False # but mask_end_prob 1 means always mask end + half others
        model_settings['mask_p'] = ((pred_len / patch_len) / ((input_dim + pred_len) // patch_len)) * 2  + 0.01 # mask future, and same amount randomly from the input
        model_settings['mask_end_prob'] = 1.0

        # reinitialize model to get code part of the model
        untrained_masked_forecast_model = Maskformer(**model_settings)
        _, masked_forecast_static = eqx.partition(untrained_masked_forecast_model, eqx.is_array)

        # join existing model parameters with new model code
        model = eqx.combine(model_params, masked_forecast_static)
        #del untrained_masked_forecast_model

        save_params_fn = lambda model, epoch, state: save_params(f'masked_forecast-{state}-e{epoch}-{args.script_name}', model, model_settings, timestamp, config.dataset_name)
        model, model_params, model_static = train_plot_save('masked_forecast', config, model_settings, train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, key=key, epochs=epochs, print_every=print_every)
        print('------------------ masked_forecast done ------------------')

    if config.forecast_epochs > 0:
        print('------------------ forecast ------------------')
        epochs = config.forecast_epochs
        optimizer = optax.adam(learning_rate=1e-3)
        key = jax.random.split(key)[0]

        # only mask end, masking percentage is just pred len patches --> normal forecasting
        model_settings['key'] = key
        model_settings['mask_end'] = True
        model_settings['mask_p'] = ((pred_len / patch_len) / ((input_dim + pred_len) // patch_len))
        model_settings['forecast'] = True

        forecast_params, forecast_static = eqx.partition(Maskformer(**model_settings), eqx.is_array)

        # In case pretrained model is missing the patch forecaster weights, manually add them
        new_model_params = eqx.tree_at(lambda t: t.forecaster_head, model_params, forecast_params.forecaster_head, is_leaf=lambda x: x is None)

        model = eqx.combine(new_model_params, forecast_static)
        save_params_fn = lambda model, epoch, state: save_params(f'forecast-{state}-e{epoch}-{args.script_name}', model, model_settings, timestamp, config.dataset_name)
        model, model_params, model_static = train_plot_save('forecast', config, model_settings, train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, key=key, epochs=epochs, print_every=print_every)
        print('------------------ forecast done ------------------')


    print('all done')



run_train()