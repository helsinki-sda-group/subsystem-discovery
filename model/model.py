#%%
import equinox as eqx
import jax.numpy as jnp
import jax
from jax import vmap
import numpy as np
from equinox.nn import MultiheadAttention, Linear, Dropout
from equinox import filter_vmap as vm
from collections.abc import Callable
import matplotlib.pyplot as plt

from model.eqx_revin import RevIN
import config

from functools import partial
@partial(jax.jit, static_argnums=(0, 1, 2))
def mask_patches(end_mask_count, total_patches, mask_count, end_mask_probability, key):
    all_patches = jnp.arange(total_patches)

    def random_strategy(_):
        return jax.random.choice(key, all_patches, shape=(mask_count,), replace=False)
    
    def mixed_strategy(_):
        specific_patches = all_patches[-end_mask_count:]  # Last N/2 patches.
        remaining_patches = jax.random.choice(key, all_patches[:-end_mask_count], shape=(mask_count-end_mask_count,), replace=False)
        return jnp.concatenate([specific_patches, remaining_patches])

    strategy = jax.random.bernoulli(key, end_mask_probability)
    masked_patches = jax.lax.cond(strategy,
                                   mixed_strategy,  # True branch
                                   random_strategy,  # False branch
                                   None)  # Operate on dummy inputs because actual logic doesn't need it

    return masked_patches

class DynamicSensorLayerNorm(eqx.Module):
    scale: jnp.ndarray  # Shape: [sensors, 1, 1]
    offset: jnp.ndarray  # Shape: [sensors, 1, 1]
    instance_norm: bool
    epsilon: float = 1e-6

    def __init__(self, num_features, instance_norm):
        self.instance_norm = instance_norm
        if instance_norm:
            self.scale = jnp.ones((1, 1, num_features))
            self.offset = jnp.zeros((1, 1, num_features))
        else:
            # Initialize scale to 1 and offset to 0 for each sensor
            self.scale = jnp.ones((num_features, 1, 1))
            self.offset = jnp.zeros((num_features, 1, 1))

    def __call__(self, x):

        if self.instance_norm:
            mean = x.mean(axis=-1, keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
            normalized = (x - mean) / jnp.sqrt(variance + self.epsilon)
            normalized = normalized * self.scale + self.offset
        else:
            # x.shape is expected to be [sensors, patches, d_model]
            mean = x.mean(axis=(1, 2), keepdims=True)
            variance = ((x - mean) ** 2).mean(axis=(1, 2), keepdims=True)
            normalized = (x - mean) / jnp.sqrt(variance + self.epsilon)
            
            # Applying the learned (initialized) scale and offset
            normalized = normalized * self.scale + self.offset
        
        return normalized


class TransformerLayerEqx(eqx.Module):
    self_attn: MultiheadAttention
    norm1: DynamicSensorLayerNorm
    norm2: DynamicSensorLayerNorm
    fc: Linear
    fc_out: Linear
    dropout: Dropout
    causal: bool

    def __init__(self, instance_norm, n_sensors, d_model, d_ff, n_heads, dropout, attn_dropout, causal, eval, *, key):
        self.self_attn = MultiheadAttention(num_heads=n_heads, query_size=d_model, dropout_p=attn_dropout, key=key, inference=eval)

        num_norm_features = d_model if instance_norm else n_sensors
        self.norm1 = DynamicSensorLayerNorm(num_norm_features, instance_norm=instance_norm)
        self.norm2 = DynamicSensorLayerNorm(num_norm_features, instance_norm=instance_norm)
        self.fc = vm(vm(Linear(in_features=d_model, out_features=d_ff, key=jax.random.split(key, 1)[0])))
        self.fc_out = vm(vm(Linear(in_features=d_ff, out_features=d_model, key=jax.random.split(key, 2)[1])))
        self.dropout = Dropout(p=dropout, inference=eval)
        self.causal = causal

    def __call__(self, x, key):
        # x is [n_sensors, patches, d_model]

        key1, key2, key3 = jax.random.split(key, 3)
        if self.causal:
            num_patches = x.shape[1]
            mask = jnp.tril(jnp.ones((num_patches, num_patches)), k=0).astype('bool')
            #mask = mask[None, None, None, :, :] # TODO won't work right now
        else:
            mask = None

        attn_output = vm(lambda x, mask, key: self.self_attn(x, x, x, mask, key=key), in_axes=(0, None, None))(x, mask, key1)
        x = x + self.dropout(attn_output, key=key2)
        x = self.norm1(x)

        fc_output = self.fc_out(jax.nn.gelu(self.fc(x)))
        x = x + self.dropout(fc_output, key=key3)
        
        x = self.norm2(x)
        return x

class PatchEncoder(eqx.Module):
    patch_tokenizer: Linear
    pos_embedding: jnp.ndarray
    layers: list
    patch_len: int

    def __init__(self, instance_norm, n_sensors, max_patch_num, patch_len, d_model, n_layers, d_ff, n_heads, dropout, attn_dropout, causal, eval, key):
        self.patch_len = patch_len
        keys = jax.random.split(key, n_layers + 2)
        self.patch_tokenizer = vm(vm(Linear(in_features=patch_len, out_features=d_model, key=keys[0]))) # mapped to each patch similarly
        self.pos_embedding = jax.random.normal(key, (max_patch_num, d_model)) - 0.5
        self.layers = [TransformerLayerEqx(instance_norm=instance_norm, n_sensors=n_sensors, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, attn_dropout=attn_dropout, causal=causal, eval=eval, key=keys[i+2]) for i in range(n_layers)]

    def __call__(self, x, indices, key):
        # [n_sensors, patches, d_model]
        n_sensors = x.shape[0]
        keys = jax.random.split(key, len(self.layers) + 1)

        x = self.patch_tokenizer(x)

        # Adding position encoding based on sequence position.
        x = x + self.pos_embedding[indices]  

        for i, layer in enumerate(self.layers):
            #key = jax.random.split(keys[i+1], (x.shape[0], x.shape[1]))
            x = layer(x, keys[i+1])
        return x

class PatchDecoder(eqx.Module):
    max_patch_num: int
    layers: list
    output_fc: Linear

    def __init__(self, instance_norm, n_sensors, patch_len, max_patch_num, d_model, n_layers, d_ff, n_heads, dropout, attn_dropout, causal, eval, key):
        self.max_patch_num = max_patch_num
        keys = jax.random.split(key, 6)
        self.layers = [TransformerLayerEqx(instance_norm=instance_norm, n_sensors=n_sensors, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, attn_dropout=attn_dropout, causal=causal, eval=eval, key=key) for _ in range(n_layers)]
        self.output_fc = vm(vm(eqx.nn.MLP(in_size=d_model, out_size=patch_len, width_size=d_ff, key=keys[3], depth=1, activation=jax.nn.leaky_relu, use_final_bias=False)))

    def __call__(self, x, key):
        keys = jax.random.split(key, len(self.layers) + 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, keys[i])

        x = self.output_fc(x)
        return x

class PatchForecaster(eqx.Module):
    total_num_patches: int
    num_input_patches: int
    patch_len: int
    pred_len: int
    d_model: int

    output_fc: Linear
    def __init__(self, mlp_depth, total_num_patches, num_input_patches, patch_len, d_model, pred_len, forecast_hidden_dim, key):
        keys = jax.random.split(key)
        self.total_num_patches = total_num_patches
        self.num_input_patches = num_input_patches
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.output_fc = vm(eqx.nn.MLP(
            in_size=d_model * num_input_patches,
            out_size=pred_len,
            width_size=forecast_hidden_dim,
            key=keys[0],
            depth=mlp_depth,
            activation=jax.nn.leaky_relu,
            use_final_bias=False
        ))
    
    def __call__(self, x):
        # input is [sensors, num_patches, d_model]
        x = x[:, :self.num_input_patches]  # take only the input patches (rest are masked)

        # flatten patches
        x = x.reshape(x.shape[0], self.num_input_patches * self.d_model)

        x = self.output_fc(x)

        x = x.reshape(x.shape[0], self.pred_len // self.patch_len, self.patch_len)

        # add zero patches to start to reach total_num_patches
        zero_patches = np.zeros((x.shape[0], self.total_num_patches - self.pred_len // self.patch_len, self.patch_len))
        x = jnp.concatenate([zero_patches, x], axis=1)

        return x
        
class SensorAggregator(eqx.Module):
    max_patch_num: int
    missing_token: jnp.ndarray
    pos_embedding: jnp.ndarray
    sensor_id_embedding: jnp.ndarray
    static_attention: jnp.ndarray
    value_proj: Linear
    univariate_norm: Callable
    interaction_norm: Callable
    adjacency_norm: Callable 
    disable_adjacency: bool

    def __init__(self, disable_adjacency, rank_div, instance_norm, n_sensors, max_patch_num, d_model, key):
        self.disable_adjacency = disable_adjacency
        self.max_patch_num = max_patch_num
        keys = jax.random.split(key, 5)
        self.missing_token = jax.random.normal(keys[0], (d_model,))
        self.pos_embedding = jax.random.normal(keys[1], (1, max_patch_num, d_model))
        self.sensor_id_embedding = jax.random.normal(keys[2], (n_sensors, 1, d_model))
        self.static_attention = jax.random.normal(keys[3], (n_sensors, n_sensors // rank_div)) * 0.0001
        self.value_proj = vm(vm(Linear(in_features=d_model, out_features=d_model, key=keys[4])))
        self.univariate_norm = vm(vm(lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)))
        self.interaction_norm = vm(vm(lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)))
        self.adjacency_norm = vm(vm(lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)))

    @staticmethod
    def norm_adjacency(adjacency):
        adjacency = jnp.abs(adjacency)
        adjacency = adjacency @ adjacency.T  # make symmetric
        adjacency -= jnp.diag(jnp.diag(adjacency))  # remove diagonal
        return adjacency

    def __call__(self, x, indices, missing_indices):
        full_x = jnp.zeros((x.shape[0], self.max_patch_num, x.shape[2]))
        full_x = vmap(lambda i, x, full_x: full_x.at[i].set(x))(indices, x, full_x)  # replace the patches at the indices with the input patches
        full_x = vmap(lambda i, full_x: full_x.at[i].set(self.missing_token))(missing_indices, full_x)  # set all idxs not in indices to missing token

        full_x += self.pos_embedding
        full_x += self.sensor_id_embedding

        if self.disable_adjacency:
            full_x = self.univariate_norm(full_x)
        else:
            static_attention = self.norm_adjacency(self.static_attention)
            sensor_interactions = self.interaction_norm(self.value_proj(full_x))
            sensor_interactions = jnp.einsum('spd,ts->tpd', sensor_interactions, static_attention)
            full_x = self.univariate_norm(full_x) + sensor_interactions  # [sensors, patches, embed]

        return full_x

class Maskformer(eqx.Module):
    encoder: PatchEncoder
    sensor_aggregator: SensorAggregator
    decoder: PatchDecoder
    forecaster_head: PatchForecaster
    revin: RevIN
    use_revin: bool
    patch_len: int
    only_forecast: bool
    mask_p: float
    mask_end_prob: float
    include_p: float
    num_include: int
    max_patch_num: int
    pred_len_patches: int
    forecast: bool

    def __init__(self, rank_div, disable_adjacency, instance_norm, revin, n_sensors, mask_end_prob, mask_p, mask_end, max_patch_num, patch_len, pred_len, d_model, n_layers, n_decoder_layers, d_ff, n_heads, dropout, attn_dropout, causal, eval, key, forecast, forecast_hidden_dim, n_forecast_layers=0):
        keys = jax.random.split(key, 4)
        self.patch_len = patch_len
        self.encoder = vm(PatchEncoder(instance_norm=instance_norm, n_sensors=n_sensors, max_patch_num=max_patch_num, patch_len=patch_len, d_model=d_model, n_layers=n_layers, d_ff=d_ff, n_heads=n_heads, dropout=dropout, attn_dropout=attn_dropout, causal=causal, eval=eval, key=keys[0]))
        self.sensor_aggregator = vm(SensorAggregator(disable_adjacency=disable_adjacency, rank_div=rank_div, instance_norm=instance_norm, n_sensors=n_sensors, max_patch_num=max_patch_num, d_model=d_model, key=keys[1]))
        self.decoder = vm(PatchDecoder(instance_norm=instance_norm, n_sensors=n_sensors, patch_len=patch_len, max_patch_num=max_patch_num, d_model=d_model, n_layers=n_decoder_layers, d_ff=d_ff, n_heads=n_heads, dropout=dropout, attn_dropout=attn_dropout, causal=causal, eval=eval, key=keys[2]))

        if forecast:
            self.forecaster_head = vm(PatchForecaster(mlp_depth=n_forecast_layers, total_num_patches=max_patch_num, num_input_patches=max_patch_num - pred_len // patch_len, patch_len=patch_len, d_model=d_model, pred_len=pred_len, forecast_hidden_dim=forecast_hidden_dim, key=keys[3]))
        else:
            self.forecaster_head = None

        self.use_revin = revin
        self.revin = RevIN(num_sensors=n_sensors, affine=self.use_revin)

        self.mask_p = mask_p
        self.mask_end_prob = mask_end_prob
        self.only_forecast = mask_end
        self.include_p = 1 - self.mask_p
        self.num_include = int(self.include_p * max_patch_num)
        self.max_patch_num = max_patch_num
        self.pred_len_patches = pred_len // patch_len
        assert pred_len % patch_len == 0, 'pred_len must be divisible by patch_len'
        if self.only_forecast:
            num_included_patches = self.max_patch_num - self.pred_len_patches
            print('included patches', num_included_patches)
            print('forecast patches', self.pred_len_patches)
        else:
            print('masking')
            print('included patches', self.num_include)
            print('mask patches', max_patch_num - self.num_include)
        
        self.forecast = forecast

    def __call__(self, x, key):
        num_patches = x.shape[2] // self.patch_len
        batch, n_sensors, _ = x.shape
        x = x.reshape(x.shape[0], x.shape[1], num_patches, self.patch_len)  # [batch, sensors, num_patches, patch_len]
        num_batches = x.shape[0]
        num_sensors = x.shape[1]

        if self.only_forecast:
            num_included_patches = self.max_patch_num - self.pred_len_patches
            included_indices = jnp.arange(num_included_patches)
            included_indices = jnp.tile(included_indices, (num_batches, num_sensors, 1))
            masked_indices = jnp.arange(num_included_patches, self.max_patch_num)
            masked_indices = jnp.tile(masked_indices, (num_batches, num_sensors, 1))
        else:
            # sample without replacement from jnp.arange(x.shape[2]) to get the position of the patches
            keys = jax.random.split(key, (num_batches, num_sensors))
            masked_indices_num = num_patches - self.num_include
            masked_indices = vm(vm(lambda key: mask_patches(self.pred_len_patches, num_patches, masked_indices_num, self.mask_end_prob, key)))(keys)
            included_indices_num = num_patches - masked_indices_num
            included_indices = vm(vm(lambda masked_indices: jnp.setdiff1d(jnp.arange(num_patches), masked_indices, size=included_indices_num)))(masked_indices)

        sensor_indices = np.arange(n_sensors)[:, None]
        x = jax.vmap(lambda x, idxs: x[sensor_indices, idxs])(x, included_indices)

        if self.use_revin:
            x, mean, std = self.revin(x, mode='norm')

        x = self.encoder(x, included_indices, jax.random.split(key, num_batches))

        x = self.sensor_aggregator(x, included_indices, masked_indices)

        if self.forecast:
            x = self.forecaster_head(x)
        else:
            x = self.decoder(x, jax.random.split(key, num_batches))

        assert x.shape == (num_batches, num_sensors, num_patches, self.patch_len), f"expected x shape {x.shape} to be {(num_batches, num_sensors, num_patches, self.patch_len)}"

        if self.use_revin:
            # Applies the mean and stdev acquired from unmasked patches on all output patches (namely the masked ones being predicted)
            x = self.revin(x, mode='denorm', mean=mean, stdev=std)

        return x, included_indices, masked_indices





dataset_specific_settings = {
    'pendulums': {
        'n_sensors': 12,
        'n_layers': 1,
        'n_decoder_layers': 1,
        'd_ff': 64,
        'd_model': 32,
        'n_heads': 4,
        'pred_len': 96,
        'patch_len': 16,
        'forecast': False,
        'forecast_hidden_dim': 2048,
    },
    'weather': {
        'n_sensors': 21,
        'n_layers': 1,
        'n_decoder_layers': 1,
        'd_ff': 64,
        'd_model': 32,
        'n_heads': 8,
        'pred_len': 96,
        'patch_len': 16,
        'forecast': False,
        'forecast_hidden_dim': 2048,
    },
    'powerplant': {
        'n_sensors': 1976,
        'n_layers': 1,
        'n_decoder_layers': 1,
        'd_ff': 64,
        'd_model': 32,
        'n_heads': 8,
        'pred_len': 96,
        'patch_len': 16,
        'forecast': False,
        'forecast_hidden_dim': 2048,
    },
    'us_weather': {
        'n_sensors': 987,
        'n_layers': 1,
        'n_decoder_layers': 1,
        'd_ff': 64,
        'd_model': 32,
        'n_heads': 8,
        'pred_len': 96,
        'patch_len': 16,
        'forecast': False,
        'forecast_hidden_dim': 2048,
    },
}

def default_mask_model_settings(dataset_name, key, args, n_sensors=None):
    max_patch_num = (args.input_dim + args.pred_len) // args.patch_len
    model_settings = {
        'disable_adjacency': False,
        'mask_p': 0.75,
        'mask_end_prob': 0.1,
        'mask_end': False, # if true, only mask patches from end of sequence up to pred_len // patch_len patches
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'causal': False,
        'eval': False,
        'key': key,
        'revin': False,
        'instance_norm': False,
        'rank_div': 1,
        'max_patch_num': max_patch_num,
        'n_forecast_layers': 0,
        **dataset_specific_settings[dataset_name]
    }
    if args is not None:
        for arg_key, value in args.items():  # vars() converts args to a dictionary
            if arg_key in model_settings:
                print('replacing', arg_key, value)
                model_settings[arg_key] = value 
            else:
                print('ignoring', arg_key, value)

    if n_sensors is not None:
        model_settings['n_sensors'] = n_sensors

    print(model_settings)
    return model_settings


def get_loss_fn(config):
    @eqx.filter_jit
    def loss_fn(model, X_batch, Y_batch, key):
        X = jnp.concatenate([X_batch, Y_batch], axis=1)

        X = jnp.transpose(X, (0, 2, 1)) # [batch, seq, n_sensors] -> [batch, sensors, seq]
        reconstruction, indices, missing_indices = model(X, key=key)
        X = X.reshape(reconstruction.shape) # patchify
        loss = jax.vmap(jax.vmap(lambda r, x, idxs: (r[idxs] - x[idxs])**2))(reconstruction, X, missing_indices)

        regularization = 0
        static_attention = SensorAggregator.norm_adjacency(model.sensor_aggregator.__wrapped__.static_attention)
        if config.eigval_reg > 0:
            # maximize eigenvalue variance
            # normalize before calculating eigenvalues to bound the eigenvalues
            scaled_static_attention = static_attention / static_attention.max()
            regularization += config.eigval_reg * -jnp.var(jnp.linalg.eigvalsh(scaled_static_attention))

        if config.l1_regularize > 0:
            regularization += config.l1_regularize * static_attention.mean()

        return jnp.mean(loss) + regularization
    
    return loss_fn
