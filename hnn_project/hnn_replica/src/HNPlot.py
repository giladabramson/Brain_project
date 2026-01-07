# Script for the generation of the plots of activity for both the tensions/currents and
# the brownian motion associated to the timescales

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from datetime import datetime
from pathlib import Path

params = {'ytick.labelsize': 20,
          'xtick.labelsize': 35,
          'axes.labelsize' : 30,
          'font.size' : 20,
          'axes.titlesize': 20}
plt.rcParams.update(params)

# Dedicated output directory per run (timestamped to avoid overwriting)
_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
OUTPUT_DIR = _OUTPUT_ROOT / _RUN_STAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, filename: str) -> Path:
    """Persist DataFrame in the per-run output directory."""
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


def _figure_path(filename: str) -> Path:
    """Provide full path for figures within the output directory."""
    return OUTPUT_DIR / filename


def _write_json(data: dict, filename: str) -> Path:
    """Serialize dictionary metadata to JSON in the output directory."""
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path

# Function for the generation of the overlap graph
def _add_stimulus_timeline(fig, stim_sequence, colors, total_time, stim_duration=None):
    if not stim_sequence:
        return
    try:
        n_segments = len(stim_sequence)
        if n_segments == 0:
            return
        fig.subplots_adjust(bottom=0.18)
        ax = fig.add_axes([0.1, 0.08, 0.82, 0.05])
        ax.set_xlim(0, total_time)
        ax.set_ylim(0, 1)
        ax.axis('off')
        segment = total_time / n_segments
        stim_len = stim_duration if stim_duration is not None else segment
        stim_len = max(0.0, min(stim_len, segment))
        t = 0
        for idx in stim_sequence:
            color = colors[idx % len(colors)]
            if stim_len > 0:
                ax.add_patch(Rectangle((t, 0), stim_len, 1, color=color, alpha=0.85))
            if stim_len < segment:
                ax.add_patch(
                    Rectangle(
                        (t + stim_len, 0),
                        segment - stim_len,
                        1,
                        color="lightgray",
                        alpha=0.25,
                    )
                )
            t += segment
        ax.text(total_time / 2, -0.3, 'Stimulus schedule (bright = driven)', ha='center', va='top', fontsize=18)
    except Exception:
        pass


def _compute_metrics(M, dt, segment_length, stim_sequence, response_threshold=0.7, retention_threshold=0.6, stim_duration=None):
    if M.ndim != 3 or segment_length <= 0:
        return pd.DataFrame()
    stim_steps = 0
    if stim_duration is not None and dt > 0:
        stim_steps = max(int(np.ceil(stim_duration / dt)), 0)
    records = []
    for seg_idx, mem_idx in enumerate(stim_sequence):
        start = seg_idx * segment_length
        end = start + segment_length
        end = min(end, M.shape[1])
        for trial in range(M.shape[2]):
            overlap = M[mem_idx, start:end, trial]
            reaction_candidates = np.where(overlap >= response_threshold)[0]
            reaction_time = reaction_candidates[0] * dt if reaction_candidates.size > 0 else np.nan

            if stim_steps >= overlap.size:
                retention_time = 0.0
            else:
                tail = overlap[stim_steps:]
                if tail.size == 0 or tail[0] < retention_threshold:
                    retention_time = 0.0
                else:
                    below_threshold = np.where(tail < retention_threshold)[0]
                    if below_threshold.size > 0:
                        retention_time = below_threshold[0] * dt
                    else:
                        retention_time = tail.size * dt

            records.append(
                {
                    "segment": seg_idx,
                    "memory": mem_idx,
                    "trial": trial,
                    "reaction_time": reaction_time,
                    "retention_time": retention_time,
                }
            )
    return pd.DataFrame.from_records(records)


def PlotOverlap(HN, EMA, y, T,col, n, sigma, In, stim_sequence=None, stim_duration=None, run_label=None, run_metadata=None):

    # Definition of the matrix for memory-trajectory overlap
    try:
        M = np.zeros((HN.P, np.size(y[0,:,0]), np.size(y[0,0,:])))
    except:
        M = np.zeros((HN.P, np.size(y[0,:,0]))) 

    # Definition of the time axis
    x = np.linspace(start=0, stop=T, num=np.size(y[0,:,0]))

    sig = np.tanh(5*y)

    # Computation of the single overlap values
    for p in range(HN.P):
        for dt in range(np.size(y[0,:,0])):
            try:
                S = np.size(y[0,0,:])
                for s in range(S):
                    M[p,dt,s] = (1/HN.N)*np.abs(np.dot(HN.mems[:,p],sig[:,dt,s]))
            except:
                M[p,dt] = (1/HN.N)*np.abs(np.dot(HN.mems[:,p],sig[:,dt]))

    try:
        M_mean = np.mean(M, axis=2)
        M_std = np.std(M, axis=2)
        M_err_1 = M_mean - M_std
        M_err_1[M_err_1<0] = 0
        M_err_2 = M_mean + M_std
        M_err_2[M_err_2>1] = 1
    except:
        pass

    # Saving of the correlation data
    if n > 1:
        if EMA.C == 'M':
            if sigma == 0:
                title = 'TrajectoriesMult_'+str(n)+'_'+In+'.csv'
            else:
                title = 'TrajectoriesMult_noise_'+str(n)+'_'+In+'.csv'
        else:
            if sigma == 0:
                title = 'TrajectoriesAdd_'+str(n)+'_'+In+'.csv'
            else:
                title = 'TrajectoriesAdd_noise_'+str(n)+'_'+In+'.csv'
    else:
        if EMA.C == 'M':
            title = 'TrajectoriesMult_PP'+str(n)+'_'+In+'.csv'
        else:
            title = 'TrajectoriesAdd_PP_'+str(n)+'_'+In+'.csv'

    M_df = pd.DataFrame(M_mean)
    _write_csv(M_df, title)

    M_df = pd.DataFrame(M_err_1)
    _write_csv(M_df, 'err1.csv')

    M_df = pd.DataFrame(M_err_2)
    _write_csv(M_df, 'err2.csv')

    stim_sequence = stim_sequence or list(range(n))
    time_steps = np.size(y[0,:,0])
    if time_steps > 1:
        dt = T / (time_steps - 1)
    else:
        dt = 1.0
    segment_length = time_steps // max(len(stim_sequence), 1)
    metrics_df = _compute_metrics(M, dt, segment_length, stim_sequence, stim_duration=stim_duration)
    if not metrics_df.empty:
        _write_csv(metrics_df, 'metrics_per_trial.csv')
        summary_df = (
            metrics_df.groupby(['segment', 'memory'])[['reaction_time', 'retention_time']]
            .agg(['mean', 'std'])
            .reset_index()
        )
        summary_df.columns = ['segment', 'memory', 'reaction_mean', 'reaction_std', 'retention_mean', 'retention_std']
        _write_csv(summary_df, 'metrics_summary.csv')

    # Optional plotting of the trajectory overlap
    SAVE_PLOTS = os.getenv('SAVE_PLOTS', '1')
    SAVE_PLOTS = False if str(SAVE_PLOTS).strip() in ['0', 'false', 'False'] else True
    if SAVE_PLOTS:
        if EMA.C == 'M':
            if sigma == 0:
                title = 'TrajectoryOverlap_Mult_'+str(n)+'_'+In+'.pdf'
            else:
                title = 'TrajectoryOverlap_Mult_noise_'+str(n)+'_'+In+'.pdf'
        elif EMA.C == 'A':
            if sigma == 0:
                title = 'TrajectoryOverlap_Add_'+str(n)+'_'+In+'.pdf'
            else:
                title = 'TrajectoryOverlap_Add_noise_'+str(n)+'_'+In+'.pdf'
        if n > 1:
            if EMA.C == 'M':
                fig = plt.figure(figsize=(35,13))
                gs = fig.add_gridspec(13,1)
                ax1 = fig.add_subplot(gs[:9,:])
                for p in range(HN.P-1,-1,-1):
                    if p>2:
                        txt = 'Other Memories'
                        ax1.plot(x[1:],M_mean[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color='orange', alpha=0.5)
                    else:
                        ax1.plot(x[1:],M_mean[p,1:], color=col[p], linewidth=6.0)
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color=col[p], alpha=0.5)   
                ax1.set_xlabel(r'$t$')
                ax1.set_ylabel(r'$(\\xi,y(t))$')
            else:
                fig = plt.figure(figsize=(35,8))
                for p in range(HN.P-1,-1,-1):
                    if p>2:
                        txt = 'Other Memories'
                        plt.plot(x[1:],M_mean[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                        plt.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color='orange', alpha=0.5)
                    else:
                        plt.plot(x[1:],M_mean[p,1:], color=col[p], linewidth=6.0)
                        plt.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color=col[p], alpha=0.5)    
                plt.ylim(bottom=0,top=1) 
                plt.xlabel(r'$t$')
                plt.ylabel(r'$(\\xi,y(t))$')
        else:
            if sigma !=0:
                fig = plt.figure(figsize=(35,13))
                gs = fig.add_gridspec(13,1)
                ax1 = fig.add_subplot(gs[:9,:])
                for p in range(HN.P-1,-1,-1):
                    if p>2:
                        txt = 'Other Memories'
                        ax1.plot(x[1:],M_mean[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color='orange', alpha=0.5)
                    else:
                        #txt = "Memory"+str(p+1)
                        ax1.plot(x[1:],M_mean[p,1:], color=col[p], linewidth=6.0)
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color=col[p], alpha=0.5)  
                ax1.set_ylim(bottom=0,top=1) 
                ax1.set_xlabel(r'$t$')
                ax1.set_ylabel(r'$(\\xi,y(t))$')
            else:
                fig = plt.figure(figsize=(35,13))
                gs = fig.add_gridspec(13,1)
                ax1 = fig.add_subplot(gs[:,:])
                for p in range(HN.P-1,-1,-1):
                    if p>2:
                        txt = 'Other Memories'
                        ax1.plot(x[1:],M_mean[p,1:], color = 'orange', alpha=0.5, linewidth=4.0, linestyle='dotted', markeredgecolor='black')
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color='orange', alpha=0.5)
                    else:
                        ax1.plot(x[1:],M_mean[p,1:], color=col[p], linewidth=6.0)
                        ax1.fill_between(x[1:], M_err_1[p,1:], M_err_2[p,1:], color=col[p], alpha=0.5)   
                ax1.set_ylim(bottom=0,top=1) 
                ax1.set_xlabel(r'$t$')
                ax1.set_ylabel(r'$(\\xi,y(t))$')
        _add_stimulus_timeline(fig, stim_sequence, col, T, stim_duration=stim_duration)
        if run_label:
            try:
                fig.suptitle(run_label, fontsize=24)
            except Exception:
                pass
        plt.savefig(_figure_path(title))
        plt.close()

    if run_metadata:
        try:
            _write_json(run_metadata, "run_metadata.json")
        except Exception:
            pass


def PlotOverlapCon(HN, y, T, d_sigma, n):

    # Definition of the overlap matrix
    M_1 = np.zeros((d_sigma, np.size(y[0,:,0,0]), np.size(y[0,0,:,0])))
    M_2 = np.zeros((d_sigma, np.size(y[0,:,0,0]), np.size(y[0,0,:,0])))
    M_3 = np.zeros((d_sigma, np.size(y[0,:,0,0]), np.size(y[0,0,:,0])))

    # Definition of the time axis
    x = np.linspace(start=0, stop=T, num=np.size(y[0,:,0,0]))

    sig = np.tanh(5*y)

    # Computation of the single overlap values
    for d in range(d_sigma):
        for dt in range(np.size(y[0,:,0,0])):
            for s in range(np.size(y[0,0,:,0])):
                M_1[d,dt,s] = (1/HN.N)*np.abs(np.dot(HN.mems[:,0],sig[:,dt,s,d]))
                M_2[d,dt,s] = (1/HN.N)*np.abs(np.dot(HN.mems[:,1],sig[:,dt,s,d]))
                M_3[d,dt,s] = (1/HN.N)*np.abs(np.dot(HN.mems[:,2],sig[:,dt,s,d]))

    
    M_mean = np.mean(M_1, axis=2)
    M_std = np.std(M_1, axis=2)
    M_err_1 = M_mean - M_std
    M_err_1[M_err_1<0] = 0
    M_err_2 = M_mean + M_std
    M_err_2[M_err_2>1] = 1

    # Saving of the file
    M_df = pd.DataFrame(M_mean)
    _write_csv(M_df, 'M_cont_1.csv')

    M_df = pd.DataFrame(M_err_1)
    _write_csv(M_df, 'err1_cont_1.csv')

    M_df = pd.DataFrame(M_err_2)
    _write_csv(M_df, 'err2_cont_1.csv')

    M_mean_1 = np.mean(M_2, axis=2)
    M_std_1 = np.std(M_2, axis=2)
    M_err_1_1 = M_mean_1 - M_std_1
    M_err_1_1[M_err_1_1<0] = 0
    M_err_2_1 = M_mean_1 + M_std_1
    M_err_2_1[M_err_2_1>1] = 1

    # Saving of the file
    M_df = pd.DataFrame(M_mean_1)
    _write_csv(M_df, 'M_cont_2.csv')

    M_df = pd.DataFrame(M_err_1_1)
    _write_csv(M_df, 'err1_cont_2.csv')

    M_df = pd.DataFrame(M_err_2_1)
    _write_csv(M_df, 'err2_cont_2.csv')

    M_mean_2 = np.mean(M_3, axis=2)
    M_std_2 = np.std(M_3, axis=2)
    M_err_1_2 = M_mean_2 - M_std_2
    M_err_1_2[M_err_1_2<0] = 0
    M_err_2_2 = M_mean_2 + M_std_2
    M_err_2_2[M_err_2_2>1] = 1

    # Saving of the file
    M_df = pd.DataFrame(M_mean_2)
    _write_csv(M_df, 'M_cont_3.csv')

    M_df = pd.DataFrame(M_err_1_2)
    _write_csv(M_df, 'err1_cont_3.csv')

    M_df = pd.DataFrame(M_err_2_2)
    _write_csv(M_df, 'err2_cont_3.csv')

    # Definitoion of the colormap
    cmap_1 = cm.get_cmap('YlOrRd')
    col_1 = cmap_1(np.linspace(0,1,d_sigma)) 

    cmap_2 = cm.get_cmap('YlGn')
    col_2 = cmap_2(np.linspace(0,1,d_sigma))

    cmap_3 = cm.get_cmap('BuPu')
    col_3 = cmap_3(np.linspace(0,1,d_sigma)) 

    col = [col_1, col_2, col_3]

    fig = plt.figure(figsize=(35,13))
    ax1 = fig.add_subplot()
    for d in range(d_sigma):      
        ax1.plot(x[500:-300],M_mean[d,500:-300], color=col[0][d], linewidth=6.0)
        ax1.fill_between(x[500:-300], M_err_1[d,500:-300], M_err_2[d,500:-300], color=col[0][d], alpha=0.1) 

        ax1.plot(x[500:-300],M_mean_1[d,500:-300], color=col[1][d], linewidth=6.0)
        ax1.fill_between(x[500:-300], M_err_1_1[d,500:-300], M_err_2_1[d,500:-300], color=col[1][d], alpha=0.1)

        ax1.plot(x[500:-300],M_mean_2[d,500:-300], color=col[2][d], linewidth=6.0)
        ax1.fill_between(x[500:-300], M_err_1_2[d,500:-300], M_err_2_2[d,500:-300], color=col[2][d], alpha=0.1)    
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'Memory Overlap')
    plt.savefig(_figure_path('ContinuityOver.pdf'), bbox_inches = 'tight', format='pdf')
    plt.close()
