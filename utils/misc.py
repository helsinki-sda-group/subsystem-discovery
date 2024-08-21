import numpy as np
import equinox as eqx
from matplotlib import animation, pyplot as plt
from tensorboardX import SummaryWriter
from datetime import datetime
from jax import numpy as jnp


def extract_weights(model, prefix=''):
    weights = {}
    if hasattr(model, '__wrapped__'):  # Check for vmapped module
        return extract_weights(model.__wrapped__, prefix)
    elif isinstance(model, eqx.Module):
        for name, child in vars(model).items():
            path = f"{prefix}.{name}" if prefix else name
            if eqx.is_inexact_array(child):
                weights[path] = child
            elif isinstance(child, eqx.Module) or hasattr(child, '__wrapped__'):
                weights.update(extract_weights(child, path))
    return weights

# %%

def animate_matrix(matrices, title, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(10,8))
    cax = ax.matshow(matrices[-1], interpolation='nearest', cmap=cmap)
    fig.colorbar(cax)

    # Custom tick labels for three double pendulums
    tick_labels = ['P1_x1', 'P1_y1', 'P1_x2', 'P1_y2', 'P2_x1', 'P2_y1', 'P2_x2', 'P2_y2', 'P3_x1', 'P3_y1', 'P3_x2', 'P3_y2']
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout(pad=3.0)

    def update(i):
        cax.set_data(matrices[i])
        ax.set_title(f'{title} | Time Step: {i+1}', pad=20)
        return cax,

    ani = animation.FuncAnimation(fig, update, frames=range(len(matrices)), interval=200, blit=False)
    plt.show()
    # Save to file
    ani.save(f'eda/pendulums/results/{title.replace(" ", "_").lower()}.gif', writer='ffmpeg', fps=15)

def log_params_tree(writer, epoch, params):
    def log_with_path(path, value):
        name = jax.tree_util.keystr(path)
        value_np = np.array(value)
        writer.add_histogram(name, value_np, epoch)

    jax.tree_util.tree_map_with_path(log_with_path, params)

import jax
def log_params(model, epoch, writer):
    params = eqx.filter(model, eqx.is_inexact_array)
    log_params_tree(writer, epoch, params)

#%%
def animate_pendulum(seq):
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    line, = ax.plot([], [], 'o-', lw=2)
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        print(f'animating frame {i}/{seq.shape[0]}')
        x1 = seq[i, 0]
        y1 = seq[i, 1]
        x2 = seq[i, 2]
        y2 = seq[i, 3]
        line.set_data([0, x1, x2], [0, y1, y2])
        return line,
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=seq.shape[0], blit=True, interval=1)
    ani.save('eda/pendulums/results/anim.mp4', writer='ffmpeg', fps=200)

    plt.show()
    return ani

def plot_trajectory(data):
    figure = plt.figure()
    pendulum_idx = 0
    plt.scatter(data[:, 0::4][:, pendulum_idx], data[:, 1::4][:, pendulum_idx], cmap='magma', label='pendulum 1', c=range(len(data)), s=0.1)
    plt.scatter(data[:, 2::4][:, pendulum_idx], data[:, 3::4][:, pendulum_idx], cmap='viridis', label='pendulum 2', c=range(len(data)), s=0.2)
    plt.show()

def get_tensorboard_writer(output_path, name=''):
    now = datetime.now()
    now_str = now.strftime('%Y%m%d_%H%M%S')
    name += '-' if name else ''
    writer = SummaryWriter(f'{output_path}/{now_str}_{name}')
    return writer

def predict_non_overlapping(model, data_loader, forecast_size):
    """Predict future states for non-overlapping windows using the model and a PyTorch DataLoader's dataset."""
    predictions = []
    ground_truths = []
    inputs = []
    adjacencies = []

    dataset = data_loader.dataset
    total_samples = 5
    
    # Iterate through the dataset in steps of forecast_size to avoid overlap
    for i, start_idx in enumerate(range(0, len(dataset), forecast_size)):
        if i > total_samples:
            break
        # Retrieve the data at the current index
        X_batch, Y_batch = dataset[start_idx]
        
        # Convert to JAX arrays; add batch dimension as model expects it
        X_batch = jnp.array(X_batch)[None]
        Y_batch = jnp.array(Y_batch)[None]
        
        # Predict
        pred, As = model(X_batch, (0, 1))
        
        predictions.append(pred)
        inputs.append(X_batch)
        ground_truths.append(Y_batch)
        adjacencies.append(As)
    
    # Concatenate lists of JAX arrays
    predictions = jnp.concatenate(predictions, axis=0)
    inputs = jnp.concatenate(inputs, axis=0)
    ground_truths = jnp.concatenate(ground_truths, axis=0)
    adjacencies = jnp.concatenate(adjacencies, axis=0)
    return inputs, predictions, ground_truths, adjacencies
