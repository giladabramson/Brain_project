# Script for the implementation of an input driven Hopfield model with 
# either only short term synaptic plasticity or both short and long
# term synaptic components

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set random seed for reproducibility across all simulations
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
np.random.seed(RANDOM_SEED)

from HNN_Gen import HNN
from Eul_May import EM
from HNPlot import PlotOverlap, PlotOverlapCon, OUTPUT_DIR

# Update of font parameters for matplotlib
params = {'ytick.labelsize': 20,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 20}
plt.rcParams.update(params)


os.system('cls')

# Define the condition of memory generation
# C = O : Orthogonal Binary
# C = R : Random Binary
C = 'R'
# Define the perturbation around one of the memories
eps = 0.5
# Scaling factors for the dynamics (close to original behavior)
w_external = float(os.getenv('W_EXTERNAL', '1.5'))
w_s = float(os.getenv('W_S', '1.0'))
# Defines the scalar amplitude of the perturbations
noise_levels = [float(os.getenv('NOISE_LEVEL', '0.5'))]
#noise_levels = [0, 2, 4]
d_sigma = len(noise_levels)

# Input configuration
input_mode = os.getenv('INPUT_MODE', 'stochastic').lower()  # 'stochastic' or 'deterministic'
deterministic_gain = float(os.getenv('DETERMINISTIC_GAIN', '2.5'))
input_noise_mode = os.getenv('INPUT_NOISE_MODE', 'constant').lower()  # 'constant' or 'per_step'
input_noise_std = float(os.getenv('INPUT_NOISE_STD', '0.0'))
input_log_trials = int(os.getenv('INPUT_LOG_TRIALS', '5'))
input_plot_trials = int(os.getenv('INPUT_PLOT_TRIALS', '3'))
input_plot_segments = int(os.getenv('INPUT_PLOT_SEGMENTS', '1'))

# Definition of the integration interval
t_ini = 0
t_end = float(os.getenv('T_END', '10'))
# Definition of the time step
dt = 0.01
# Duration (in model seconds) for which the external stimulus acts
stim_duration = float(os.getenv('STIM_DURATION', '2.0'))

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)
stim_mask = x < stim_duration
stim_steps = np.count_nonzero(stim_mask)

# Generation of the Hopfield model
HN = HNN(C,eps)
# Generation of the memories
HN.net()
# Generation of the initial condition
HN.y0_gen()
# Storing of the original value 
y_buff = HN.y0

# Define the length of the input sequence
n = int(os.getenv('NUM_SEGMENTS', '1'))
# Define which memory index each segment targets (wrap if n > HN.P)
stim_sequence = [idx % HN.P for idx in range(n)]
# Define the number of samples
S = int(os.getenv('TRIAL_SAMPLES', '50'))

# Definition of the inputs
# Definition of the inputs (mirroring original settings)
u_proto = np.random.uniform(low=0.8,high=1.5,size=(HN.P, n, S))
for s in range(S):
    for j in range(n):
        u_proto[j,j,s] = np.random.uniform(low=2.0,high=3.5)
        if j>0:
             u_proto[j-1,j,s] = np.random.uniform(low=0.05,high=0.2)

# Definition of the colormap for the weights
base_colors = ['red', 'green', 'blue', 'black']
col = ['cyan' for _ in range(HN.P)]
for idx, color in enumerate(base_colors):
    if idx < HN.P:
        col[idx] = color

input_profiles = np.zeros((HN.N, T, n, S))
input_log_records = []

def _build_input_vector(gain_vector):
    return HN.mems @ gain_vector

for s in range(S):
    for h in range(n):
        mem_idx = stim_sequence[h]
        gain_vector = np.zeros(HN.P)
        if input_mode == 'deterministic':
            gain_vector[mem_idx] = deterministic_gain
        else:
            gain_vector = u_proto[:, h, s]
        u_vec = _build_input_vector(gain_vector) if input_mode != 'deterministic' else deterministic_gain * HN.mems[:, mem_idx]
        profile = np.zeros((HN.N, T))
        if stim_steps > 0:
            if input_noise_mode == 'per_step':
                noise = input_noise_std * np.random.randn(HN.N, stim_steps)
                profile[:, stim_mask] = u_vec[:, None] + noise
            else:
                noise_vec = input_noise_std * np.random.randn(HN.N)
                profile[:, stim_mask] = u_vec[:, None] + noise_vec[:, None]
        input_profiles[:, :, h, s] = profile
        base_overlap = (1/HN.N)*np.dot(HN.mems[:, mem_idx], u_vec)
        record = {
            "trial": s,
            "segment": h,
            "memory": mem_idx,
            "input_mode": input_mode,
            "noise_mode": input_noise_mode,
            "noise_std": input_noise_std,
            "base_overlap": base_overlap,
        }
        for idx_mem in range(HN.P):
            record[f"gain_mem_{idx_mem}"] = gain_vector[idx_mem] if idx_mem < gain_vector.shape[0] else 0.0
        if s < input_log_trials:
            input_log_records.append(record)

# Definition of the entire solution vector
# IDP Hopfield
Y = np.zeros((HN.N,T*n,S, d_sigma))
# Classic Hopfield with modulated stimulus
Y_add = np.zeros((HN.N,T*n,S, d_sigma))
# Classic Hopfield with constant stimulus
Y_add_c = np.zeros((HN.N,T*n,S, d_sigma))


for s in range(S):
    # Generation of the initial condition
    y_buff = np.random.randn(HN.N)
    for d in range(d_sigma):
        HN.y0 = y_buff
        for j in range(n):
            # Generation of the Euler-Mayorama Integrator for the SDE
            # EMa = EM(HN, t_ini, t_end, dt, 'M','c', w_external=w_external, w_s=w_s)                 # SDP Hopfield
            EMa_add = EM(HN, t_ini, t_end, dt, 'A','m', w_external=w_external, w_s=w_s, stim_duration=stim_duration)             # Classic Hopfield - Modulated Stimulus
            #EMa_add_c = EM(HN, t_ini, t_end, dt, 'A','c', w_external=w_external, w_s=w_s)           # Classic Hopfield - Constant Stimulus

            # Integration of the system over the time interval
            # IDP Hopfield - Uncomment if needed
            # if j>0:
            #     HN.y0 = y[:,-1]
            # y = EMa.Eu_Ma_Test(HN, noise_levels[d], u_tran[:,j,s])

            # Classic Hopfield w modulated input - Uncomment if needed
            if j>0:
               HN.y0 = y_add[:,-1]
            y_add = EMa_add.Eu_Ma_Test(HN, noise_levels[d], input_profiles[:, :, j, s])

            # Classic Hopfield w constant input - Uncomment if needed
            #if j>0:
            #    HN.y0 = y_add_c[:,-1]
            #y_add_c = EMa_add_c.Eu_Ma_Test(HN, noise_levels[d], u_tran[:,j,s])

            # Update of the trajectories
            # Y[:,j*T:(j+1)*T, s, d] = y
            Y_add[:,j*T:(j+1)*T, s, d] = y_add
            #Y_add_c[:,j*T:(j+1)*T, s,d] = y_add_c

# Plotting of the overlap during training
if d_sigma == 1:
    # PlotOverlap(HN, EMa, Y, n*t_end, col, n, noise_levels, 'c')
    base_noise = noise_levels[0] if noise_levels else 0.0
    run_label = None
    if base_noise is not None:
        run_label = f"w_s={w_s:.2f}, w_ext={w_external:.2f}, noise={base_noise:.2f}"
    run_metadata = {
        "w_s": w_s,
        "w_external": w_external,
        "noise_levels": noise_levels,
        "stim_duration": stim_duration,
        "segment_count": n,
        "segment_duration": t_end,
        "dt": dt,
        "stim_sequence": stim_sequence,
    }
    PlotOverlap(
        HN,
        EMa_add,
        Y_add,
        n*t_end,
        col,
        n,
        base_noise,
        'm',
        stim_duration=stim_duration,
        stim_sequence=stim_sequence,
        run_label=run_label,
        run_metadata=run_metadata,
    )
    #PlotOverlap(HN, EMa_add_c, Y_add_c, n*t_end, col, n, noise_levels, 'c')
else:
    PlotOverlapCon(HN, Y, n*t_end, d_sigma, n)

# Persist input sampling log
if input_log_records:
    df_inputs = pd.DataFrame(input_log_records)
    df_inputs.to_csv(OUTPUT_DIR / 'input_gains_log.csv', index=False)

def plot_input_alignment(HN, profiles, stim_sequence, max_trials, max_segments, time_axis, stim_duration):
    trials_to_plot = min(max_trials, profiles.shape[3])
    for s in range(trials_to_plot):
        for j in range(min(profiles.shape[2], max_segments)):
            mem_idx = stim_sequence[j]
            mem_vec = HN.mems[:, mem_idx]
            profile = profiles[:, :, j, s]
            overlap = (1/HN.N)*np.dot(mem_vec, profile)
            fig, axes = plt.subplots(2, 1, figsize=(18, 10), dpi=120)
            axes[0].plot(time_axis, overlap, color='tab:blue', linewidth=3)
            axes[0].axvspan(0, stim_duration, color='red', alpha=0.15)
            axes[0].set_ylabel('I(t)·ξ / N')
            axes[0].set_title(f'Trial {s} – Segment {j} – Memory {mem_idx}')
            axes[0].set_xlabel('Time')
            im = axes[1].imshow(profile, aspect='auto', extent=[time_axis[0], time_axis[-1], 0, HN.N], cmap='coolwarm')
            axes[1].set_ylabel('Neuron index')
            axes[1].set_xlabel('Time')
            axes[1].set_title('Input drive per neuron')
            axes[1].axvspan(0, stim_duration, color='red', alpha=0.1)
            fig.colorbar(im, ax=axes[1], orientation='vertical', label='Input amplitude')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'input_profile_trial{s}_segment{j}.png')
            plt.close(fig)

plot_input_alignment(HN, input_profiles, stim_sequence, input_plot_trials, input_plot_segments, x, stim_duration)


####################################################################################################
# Glitch experiment - uncomment if needed

# # Definition of the integration interval
# t_ini = 0
# t_end = 25

# # Generation of the single input window
# x = np.arange(start=0,stop=t_end,step=dt)
# T = np.size(x)

# # Definition of the input
# u_tot = np.zeros((HN.N,T,S))
# for s in range(S):
#     u_tot[:,:800, s] = np.diag(u_tran[:,0, s])@np.ones((HN.N,800))
#     u_tot[:,800:1000, s] = np.diag(u_tran[:,1, s])@np.ones((HN.N,200))
#     u_tot[:,1000:1500, s] = np.diag(u_tran[:,0, s])@np.ones((HN.N,500))
#     u_tot[:,1500:1900, s] = np.diag(u_tran[:,1, s])@np.ones((HN.N,400))
#     u_tot[:,1900:2500, s] = np.diag(u_tran[:,2, s])@np.ones((HN.N,600))


# # Re-definition of n for the continuous input
# n = 1

# # Definition of the entire solution vector
# Y = np.zeros((HN.N,T*n,S))
# Y_add = np.zeros((HN.N,T*n,S))
# HN.y0 = y_buff

# for s in range(S):
#     HN.y0 = np.random.randn(HN.N)
#     for j in range(n):
#         # Generation of the Euler-Mayorama Integrator for the SDE
#         EMa = EM(HN, t_ini, t_end, dt, 'M', 'c', w_external=w_external, w_s=w_s)
#         EMa_add = EM(HN, t_ini, t_end, dt, 'A', 'm', w_external=w_external, w_s=w_s)
#         # Integration of the system over the time interval
#         y = EMa.Eu_Ma_Test(HN, noise_levels[0], u_tot[:,:, s])
#         y_add = EMa_add.Eu_Ma_Test(HN, noise_levels[0], u_tot[:,:,s])

#     Y[:,j*T:(j+1)*T, s] = y
#     Y_add[:,j*T:(j+1)*T, s] = y_add


# # Plotting of the overlap during training
# PlotOverlap(HN, EMa, Y, n*t_end, col, n, noise_levels, 'c')
# PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, noise_levels, 'm')
