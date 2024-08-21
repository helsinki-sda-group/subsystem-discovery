import jax.numpy as jnp
import jax
import numpy as np
from matplotlib import pyplot as plt
def plot_pred_with_masking(x, y, model, batch_idx, filename, patch_len=16, ADJ_OUTPUT_PATH='output'):
    print('plotting preds')

    sensor_idx = 0
    inp = jnp.concatenate([x, y], axis=1)
    inp = jnp.transpose(inp, (0, 2, 1))  # [batch, seq, n_sensors] -> [batch, n_sensors, seq]
    pred, indices, missing_indices = model(inp, key=jax.random.PRNGKey(1337))
    pred = pred.reshape(inp.shape)  # [batch, sensors, patches, patchlen] -> [batch, sensors, seq]

    fig = plt.figure(figsize=(10, 5))
    
    # Plot input, target, and prediction
    seq_length = x.shape[1] + y.shape[1]
    t = np.arange(seq_length)
    plt.plot(t, inp[batch_idx, sensor_idx, :], label='Target', color='green')

    # Color patches
    all_indices = np.arange(seq_length // patch_len)
    mask = np.isin(all_indices, missing_indices[batch_idx, sensor_idx])

    # Plot only the masked parts for predictions
    for i, masked in enumerate(mask):
        if masked:
            if i == 0:
                plt.plot(t[i*patch_len:(i+1)*patch_len], pred[batch_idx, sensor_idx, i*patch_len:(i+1)*patch_len], label='Prediction', color='blue')
            else:
                plt.plot(t[i*patch_len:(i+1)*patch_len], pred[batch_idx, sensor_idx, i*patch_len:(i+1)*patch_len], color='blue')
    #plt.plot(t, pred[batch_idx, sensor_idx, :], label='Prediction', color='blue')

    masked_color = 'red'
    original_color = 'lightgrey'
    for i, masked in enumerate(mask):
        if masked:
            plt.axvspan(i*patch_len, (i+1)*patch_len, color=masked_color, alpha=0.1)
        else:
            plt.axvspan(i*patch_len, (i+1)*patch_len, color=original_color, alpha=0.1)

    first_legend = plt.legend(loc='lower right')  # Save the first legend
    plt.gca().add_artist(first_legend)  # Add the first legend back after creating the second

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=masked_color, edgecolor=masked_color, alpha=0.3, label='Masked (Predicted)'),
                       Patch(facecolor=original_color, edgecolor=original_color, alpha=0.3, label='Unmasked (Input)')]
    plt.legend(handles=legend_elements, loc='upper left')

    plt.savefig(f'{ADJ_OUTPUT_PATH}/{filename}.png')
    plt.close(fig)
    plt.clf()

def plot_adjacency(adjacency, filepath):
    print('plot adjacency')
    adjacency = np.abs(adjacency)
    fig, ax = plt.subplots(figsize=(10,8))
    cax = ax.matshow(adjacency, interpolation='nearest', cmap='viridis')
    fig.colorbar(cax)

    if adjacency.shape == (12, 12):
        # Custom tick labels for three double pendulums
        tick_labels = ['P1_x1', 'P1_y1', 'P1_x2', 'P1_y2', 'P2_x1', 'P2_y1', 'P2_x2', 'P2_y2', 'P3_x1', 'P3_y1', 'P3_x2', 'P3_y2']
        ax.set_xticks(range(len(tick_labels)))
        ax.set_yticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=90)
        ax.set_yticklabels(tick_labels)
    plt.tight_layout(pad=3.0)
    plt.savefig(filepath)