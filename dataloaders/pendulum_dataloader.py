import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from numpy import cos, sin


# Define constants
G = 9.8  # Acceleration due to gravity, in m/s^2
L1 = 2.0  # Length of pendulum 1 in m
L2 = 1.0  # Length of pendulum 2 in m
M1 = 1.0  # Mass of pendulum 1 in kg
M2 = 1.0  # Mass of pendulum 2 in kg
params = [L1, L2, M1, M2, G]

def get_data(initial_states, t, dt, t_stop, params):
    L1, L2, M1, M2, G = params
    damped = False
    # Differential equations for the double pendulum
    def derivs(t, state):
        if damped:
            θ1, ω1, θ2, ω2 = state
            delta = θ2 - θ1
            den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
            den2 = (L2 / L1) * den1
            
            # Damping coefficients
            b1 = 0.05  # Damping coefficient for the first pendulum
            b2 = 0.05  # Damping coefficient for the second pendulum

            dydx = np.zeros_like(state)
            dydx[0] = ω1
            
            dydx[1] = ((M2 * L1 * ω1**2 * sin(delta) * cos(delta) +
                        M2 * G * sin(θ2) * cos(delta) +
                        M2 * L2 * ω2**2 * sin(delta) -
                        (M1 + M2) * G * sin(θ1)) / den1 -
                    b1 * ω1)  # Added damping term for the first pendulum
            
            dydx[2] = ω2
            
            dydx[3] = ((-M2 * L2 * ω2**2 * sin(delta) * cos(delta) +
                        (M1 + M2) * G * sin(θ1) * cos(delta) -
                        (M1 + M2) * L1 * ω1**2 * sin(delta) -
                        (M1 + M2) * G * sin(θ2)) / den2 -
                    b2 * ω2)  # Added damping term for the second pendulum

            return dydx
        else:
            dydx = np.zeros_like(state)

            dydx[0] = state[1]

            delta = state[2] - state[0]
            den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
            dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                        + M2 * G * sin(state[2]) * cos(delta)
                        + M2 * L2 * state[3] * state[3] * sin(delta)
                        - (M1+M2) * G * sin(state[0]))
                    / den1)

            dydx[2] = state[3]

            den2 = (L2/L1) * den1
            dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                        + (M1+M2) * G * sin(state[0]) * cos(delta)
                        - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                        - (M1+M2) * G * sin(state[2]))
                    / den2)

            return dydx

    # Solve the ODE for each pendulum
    tol = 1e-7
    solutions = [solve_ivp(derivs, [0, t_stop], state, t_eval=t, rtol=tol, atol=tol) for state in initial_states]
    print([sol.message for sol in solutions])
    solutions = [sol.y for sol in solutions]
    return solutions

def plot_pendulum(solutions, initial_states, t, dt):
    # Set up the figure and axes for the pendulum animation and phase space plots
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    # Initialize lines for pendulum animation and phase space plots
    lines = []
    points = []
    time_texts = []
    for idx, ax_row in enumerate(axs):
        ax = ax_row
        ax.set_xlim(-2 * L1, 2 * L1)
        ax.set_ylim(-2 * L1, L1)
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        point, = ax.plot([], [], 'ro')
        lines.append(line)
        points.append(point)

        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        time_texts.append(time_text)

    # Initialize the phase space history

    # Animation function
    def animate(i):
        for idx, sol in enumerate(solutions):
            x1 = L1 * sin(sol[0, i])
            y1 = -L1 * cos(sol[0, i])
            x2 = L2 * sin(sol[2, i]) + x1
            y2 = -L2 * cos(sol[2, i]) + y1

            lines[idx].set_data([0, x1, x2], [0, y1, y2])
            points[idx].set_data([x2], [y2])

            time_texts[idx].set_text(f'time = {i * dt:.1f}s')

        # Return a flat list of artists that have changed for blitting
        return [item for sublist in zip(lines, points, time_texts) for item in sublist]

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=dt*500, blit=True)
    return ani

def get_coordinate_data(initial_states, t_eval, dt, t_stop, params):
    solutions = get_data(initial_states, t_eval, dt, t_stop, params)
    print(solutions[0].shape)
    print(solutions[1].shape)
    print(solutions[2].shape)
    assert solutions[0].shape == solutions[1].shape == solutions[2].shape
    #ani = plot_pendulum(solutions, initial_states, t_eval, dt)
    #plt.show()
    # ani.save('eda/pendulums/triple_pendulum.mp4', writer='ffmpeg', fps=100)
    #%%
    angle_data = np.concatenate(solutions, axis=0)
    angle_data = angle_data.T
    angle_data.shape # [1000, 12]
    #%%
    # Lets make the data consist only of the x and y positions of the pendulum joints without knowledge of speed
    # every 4 cols are a new pendulum
    x1s = L1 * np.sin(angle_data[:, 0::4])
    y1s = -L1 * np.cos(angle_data[:, 0::4])
    x2s = L2 * np.sin(angle_data[:, 2::4]) + x1s
    y2s = -L2 * np.cos(angle_data[:, 2::4]) + y1s

    #data = jnp.concatenate([x1s, y1s, x2s, y2s], axis=1)
    # instead concatenate so that each pendulums x1, y1, x2, y2 are together, so not like above lien
    # so [x1p1, y1p1, x2p1, y2p1, x1p2, y1p2, x2p2, y2p2, x1p3, y1p3, x2p3, y2p3]
    # data = jnp.concatenate([x1s, y1s, x2s, y2s], axis=1)# this won't work, this will organize them like [x1p1, x1p2, x1p3, y1p1, y1p2, y1p3, x2p1, x2p2, x2p3, y2p1, y2p2, y2p3]
    data = np.concatenate([x1s[:, 0:1], y1s[:, 0:1], x2s[:, 0:1], y2s[:, 0:1], 
                            x1s[:, 1:2], y1s[:, 1:2], x2s[:, 1:2], y2s[:, 1:2], 
                            x1s[:, 2:3], y1s[:, 2:3], x2s[:, 2:3], y2s[:, 2:3]], axis=1)

    #%%
    data.shape
    return data

# Training visualization etc
def split_data(data, train_frac=0.8, val_frac=0.10, test_frac=0.10):
    assert train_frac + val_frac + test_frac == 1.0
    n = data.shape[0]
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

# Function to create sequences of length L for predicting F future steps
def create_sequences(data, L, F):
    sequences = []
    targets = []
    for start in range(len(data) - L - F):
        end = start + L
        sequences.append(data[start:end])
        targets.append(data[end:end + F])
    return np.array(sequences), np.array(targets)

def create_dataloader(X, Y, batch_size, shuffle):
    X_torch = torch.tensor(X, dtype=torch.float32)
    Y_torch = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_torch, Y_torch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_dataloaders(data, lookback_size, forecast_size, batch_size):
    data_np = np.array(data)
    assert lookback_size + forecast_size < len(data_np), "L + F cannot be greater than the length of the data"
    train_data, val_data, test_data = split_data(data_np)

    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True)

    train_data = (train_data - train_mean) / train_std
    val_data = (val_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    train_loader = create_dataloader(*create_sequences(train_data, lookback_size, forecast_size), batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(*create_sequences(val_data, lookback_size, forecast_size), batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(*create_sequences(test_data, lookback_size, forecast_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_default_pendulum_data(lookback_size=128, forecast_size=128, batch_size=4096, t_stop=1_00, dt=0.01):
    #%%
    # The first value is the initial angle of the upper pendulum arm (θ1θ1​).
    # The second value is the initial angular velocity of the upper pendulum arm (ω1ω1​).
    # The third value is the initial angle of the lower pendulum arm (θ2θ2​).
    # The fourth value is the initial angular velocity of the lower pendulum arm (ω2ω2
    # Initial states for three different pendulums
    initial_states = [
        np.radians([91.0, 0.0, -10.0, 0.0]),
        np.radians([270.0, 0.0, -1., 0.0]),
        np.radians([181.0, 0.0, -5.0, 0.0]),
    ]

    # How many seconds to simulate
    t_eval = np.arange(0, t_stop, dt)
    #%%
    G = 9.8  # Acceleration due to gravity, in m/s^2
    L1 = 2.0  # Length of pendulum 1 in m
    L2 = 1.0  # Length of pendulum 2 in m
    M1 = 1.0  # Mass of pendulum 1 in kg
    M2 = 1.0  # Mass of pendulum 2 in kg
    simulation_params = [L1, L2, M1, M2, G]

    #%%
    print('simulating data')
    data = get_coordinate_data(initial_states, t_eval, dt, t_stop, simulation_params)
    #%%
    #%%
    ####################################
    ### Create dataloaders
    ####################################

    train_loader, val_loader, test_loader = get_dataloaders(data, lookback_size, forecast_size, batch_size)
    return train_loader, val_loader, test_loader, data

def dump_to_text_file():
    t_stop = 1000
    dt = 0.025
    initial_states = [
        np.radians([91.0, 0.0, -10.0, 0.0]),
        np.radians([270.0, 0.0, -1., 0.0]),
        np.radians([181.0, 0.0, -5.0, 0.0]),
    ]
    # How many seconds to simulate
    t_eval = np.arange(0, t_stop, dt)
    #%%
    G = 9.8  # Acceleration due to gravity, in m/s^2
    L1 = 2.0  # Length of pendulum 1 in m
    L2 = 1.0  # Length of pendulum 2 in m
    M1 = 1.0  # Mass of pendulum 1 in kg
    M2 = 1.0  # Mass of pendulum 2 in kg
    simulation_params = [L1, L2, M1, M2, G]

    print('simulating')
    data = get_coordinate_data(initial_states, t_eval, dt, t_stop, simulation_params)
    print('done simulating')
    np.savetxt('output/tmp/pendulum.txt', data, delimiter=',', fmt='%1.5f')

#%%
if __name__ == "__main__":
    dump = True
    if dump:
        dump_to_text_file()
        import os
        exit()

    
    t_stop = 5 # How many seconds to simulate
    # How many trajectory points to display
    # Time array
    dt = 0.01
    t = np.arange(0, t_stop, dt)

    # The first value is the initial angle of the upper pendulum arm (θ1θ1​).
    # The second value is the initial angular velocity of the upper pendulum arm (ω1ω1​).
    # The third value is the initial angle of the lower pendulum arm (θ2θ2​).
    # The fourth value is the initial angular velocity of the lower pendulum arm (ω2ω2
    # Initial states for three different pendulums
    initial_states = [
        np.radians([120.0, 0.0, -10.0, 0.0]),
        np.radians([121.0, 0.0, -11.0, 0.0]),
        np.radians([122.0, 0.0, -12.0, 0.0]),
    ]

    solutions = get_data(initial_states, t, dt, t_stop, params)
    anim = plot_pendulum(solutions, initial_states, t, dt)
    plt.show()
# %%