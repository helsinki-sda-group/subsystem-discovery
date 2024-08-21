import equinox as eqx
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np

def count_params(model):
    all_params = eqx.filter(model, eqx.is_inexact_array)
    print('total params', sum(x.size for x in jax.tree_util.tree_leaves(all_params)))

# multidevice stuff ---------------
num_jax_devices = len(jax.devices()) # make jittable

#@eqx.filter_jit
def split(arr):
    """Splits the first axis of `arr` evenly across the number of devices."""
    # if not divisible drop the remainder
    if arr.shape[0] % num_jax_devices != 0:
        arr = arr[:arr.shape[0] - (arr.shape[0] % num_jax_devices)]
    return arr.reshape(num_jax_devices, arr.shape[0] // num_jax_devices, *arr.shape[1:])

#@eqx.filter_jit
def get_single_device_model(model):
    if num_jax_devices == 1:
        return model # no op hopefully

    params, static = eqx.partition(model, eqx.is_inexact_array)
    single_params = jax.tree_map(lambda x: x[0], params)
    return join_params(single_params, static)

@eqx.filter_jit
def join_params(params, static):
    return eqx.combine(params, static)

@eqx.filter_jit
def make_step(model, x, y, opt_state, optimizer, loss_fn, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

from functools import partial
@eqx.filter_pmap(axis_name='num_devices', in_axes=(0, 0, 0, 0, None, None, 0))
def _make_step_multigpu(model, x, y, opt_state, optimizer, loss_fn, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices') # just for logging

    params, static = eqx.partition(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

@eqx.filter_jit
def make_step_multigpu(model, x, y, opt_state, optimizer, loss_fn, key):
    x, y = split(x), split(y)
    keys = jax.random.split(key, num_jax_devices)
    return _make_step_multigpu(model, x, y, opt_state, optimizer, loss_fn, keys)

from model.model import SensorAggregator
from utils.plot import plot_adjacency, plot_pred_with_masking

def train_plot_save(run_type, config, model_settings, train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, key, epochs, print_every):
    model_type_name = f'm{config.epochs}-mf{config.masked_forecast_epochs}-f{config.forecast_epochs}-da{1 if config.disable_adjacency else 0}'

    model = train_and_evaluate(train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params_fn, run_type=run_type, key=key, epochs=epochs, print_every=print_every)

    save_params_fn(model, epochs, 'last')
    print(f'{model_type_name}-{run_type} training done')
    static_attention = SensorAggregator.norm_adjacency(model.sensor_aggregator.__wrapped__.static_attention)

    if not config.disable_adjacency:
        plot_adjacency(static_attention, f'{config.adj_output_path}/{config.timestamp}-{model_type_name}-{run_type}-adjacency-{config.script_name}.png')
    x, y, *_ = next(iter(test_loader))
    plot_pred_with_masking(x.numpy()[:3], y.numpy()[:3], model, 0, f'{config.timestamp}-{model_type_name}-{run_type}-pred-{config.script_name}', patch_len=config.patch_len, ADJ_OUTPUT_PATH=config.adj_output_path)
    print(f'{run_type} training / plotting done')
    model_params, model_static = eqx.partition(model, eqx.is_array)
    return model, model_params, model_static

def train_and_evaluate(train_loader, val_loader, test_loader, model, optimizer, writer, loss_fn, evaluate, log_params, save_params, key, run_type, epochs=1, print_every=100):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    print(f'{run_type} val loss before training', evaluate(val_loader, model, loss_fn, key))
    log_params(model, 0, writer)

    global make_step
    n_devices = len(jax.devices())
    if n_devices > 1:
        params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
        opt_state = jax.tree_map(lambda x: jnp.array([x] * n_devices), opt_state)
        make_step = make_step_multigpu
        model = eqx.combine(params, static)

    key, subkey = jax.random.split(key)
    best_val_loss = jnp.inf
    best_epoch = 0
    best_path = ''

    max_epochs_for_logging_every_epoch = 1000
    log_params_every = 1
    if epochs > max_epochs_for_logging_every_epoch:
        log_params_every = epochs // 100
    for epoch in range(epochs):
        train_loss = []
        # Training loop
        for batch_idx, (X_batch, Y_batch, *_) in enumerate(tqdm(train_loader)):
            X_batch = jnp.array(X_batch.numpy())
            Y_batch = jnp.array(Y_batch.numpy())
            key, subkey  = jax.random.split(key)

            #X_batch, Y_batch = jax.device_put((X_batch, Y_batch), shard)
            # TODO autoparallelization doesn't work, reserves too much memory
            #keys = jax.random.split(key, (num_devices, 1, 1))
            #subkey = jax.device_put(keys, shard.reshape((num_devices, 1, 1, 1)))

            loss, model, opt_state = make_step(model, X_batch, Y_batch, opt_state, optimizer, loss_fn, subkey)
            train_loss.append(loss)
            # Print loss every so often
            if (batch_idx + 1) % print_every == 0 and batch_idx != epochs - 1:
                print(f"Epoch {epoch}, Batch {batch_idx + 1}, Loss: {np.mean(train_loss)}")
                train_loss = []  # Reset train loss

                val_loss = evaluate(val_loader, get_single_device_model(model), loss_fn, key)
                print(f"Epoch {epoch}, Validation Loss: {np.mean(val_loss)}")
        
        if epoch % log_params_every == 0:
            log_params(get_single_device_model(model), epoch + 1, writer)

        val_loss = evaluate(val_loader, get_single_device_model(model), loss_fn, key)
        print(f"Epoch {epoch}, Validation Loss: {np.mean(val_loss)}")
        if val_loss < best_val_loss:
            print(f"{run_type} best val epoch {epoch}, saving model")
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = save_params(get_single_device_model(model), 0, 'best')
        writer.add_scalar(f'{run_type}-val_loss', np.mean(val_loss), epoch +1)

    test_loss = evaluate(test_loader, get_single_device_model(model), loss_fn, key)
    print(f'{run_type} last epoch test loss', test_loss)
    writer.add_scalar(f'{run_type}-last_test_loss', test_loss, epoch)
    
    best_model = eqx.tree_deserialise_leaves(best_path, get_single_device_model(model))
    test_loss = evaluate(test_loader, best_model, loss_fn, key)
    print(f'{run_type} best epoch {best_epoch} test loss', test_loss)
    writer.add_scalar(f'{run_type}-best_test_loss', test_loss, best_epoch)

    return best_model

def evaluate_multigpu(loader, model, loss_fn, key):
    loss = []
    for X_batch, Y_batch, *_ in loader:
        key, subkey = jax.random.split(key)
        X_batch = jnp.array(X_batch.numpy())
        Y_batch = jnp.array(Y_batch.numpy())
        loss.append(loss_fn(model, X_batch, Y_batch, key))
    return np.mean(loss)

def evaluate(loader, model, loss_fn, key):
    loss = []
    for X_batch, Y_batch, *_ in loader:
        key, subkey = jax.random.split(key)
        X_batch = jnp.array(X_batch.numpy())
        Y_batch = jnp.array(Y_batch.numpy())
        loss.append(loss_fn(model, X_batch, Y_batch, key))
    return np.mean(loss)


@eqx.filter_pmap(axis_name='i', in_axes=(0, None, 0, 0, 0))
def pmap_loss_fn(model, loss_fn, X_batch, Y_batch, key):
    # Assuming loss_fn is adapted to work with pmap, handling batched inputs.
    loss = loss_fn(model, X_batch, Y_batch, key)
    return jax.lax.pmean(loss, axis_name='i')

def evaluate_multigpu(loader, model, loss_fn, key):
    losses = []
    for X_batch, Y_batch, *_ in loader:
        keys = jax.random.split(key, num=num_jax_devices + 1)
        key, subkeys = keys[0], keys[1:]

        X_batch = jnp.array(X_batch.numpy())
        Y_batch = jnp.array(Y_batch.numpy())
        X_split = split(X_batch)
        Y_split = split(Y_batch)

        batch_losses = pmap_loss_fn(model, loss_fn, X_split, Y_split, subkeys)
        losses.append(batch_losses)

    # Mean loss across all batches and devices
    #return np.mean(jnp.concatenate(losses))
    return np.mean(losses)
