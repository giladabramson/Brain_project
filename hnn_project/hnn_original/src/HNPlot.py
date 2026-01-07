# Script for the generation of the plots of activity for both the tensions/currents and
# the brownian motion associated to the timescales

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
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

# Function for the generation of the overlap graph
def PlotOverlap(HN, EMA, y, T,col, n, sigma, In):

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

    # Plotting of the trajectory overlap
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
            ax1.set_ylabel(r'$(\xi,y(t))$')
            ax2 = fig.add_subplot(gs[11:,:])
            ax2.plot(x[1:1000],0.15*np.ones((999,))+0.01*np.random.randn(999),color=col[0], linewidth=10.0)
            ax2.plot(x[1000:2000],0.15*np.ones((1000,))+0.01*np.random.randn(1000),color=col[1],linewidth=10.0)
            ax2.plot(x[2000:3000],0.15*np.ones((1000,))+0.01*np.random.randn(1000),color=col[2],linewidth=10.0)
            ax2.set_ylim(bottom=0,top=0.3)
            ax2.set_axis_off()
            ax2.set_title('Dominant Memory', fontweight='bold')
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
            plt.ylabel(r'$(\xi,y(t))$')
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
            ax1.set_ylabel(r'$(\xi,y(t))$')
            ax2 = fig.add_subplot(gs[11:,:])
            ax2.plot(x[1:800],0.15*np.ones((799,))+0.01*np.random.randn(799),color=col[0], linewidth=10.0)
            ax2.plot(x[800:1000],0.15*np.ones((200,))+0.01*np.random.randn(200),color=col[1],linewidth=10.0)
            ax2.plot(x[1000:1500],0.15*np.ones((500,))+0.01*np.random.randn(500),color=col[0],linewidth=10.0)
            ax2.plot(x[1500:1900],0.15*np.ones((400,))+0.01*np.random.randn(400),color=col[1],linewidth=10.0)
            ax2.plot(x[1900:2500],0.15*np.ones((600,))+0.01*np.random.randn(600),color=col[2],linewidth=10.0)
            ax2.set_ylim(bottom=0,top=0.3)
            ax2.set_axis_off()
            ax2.set_title('Dominant Memory', fontweight='bold')
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
            ax1.set_ylabel(r'$(\xi,y(t))$')
    plt.savefig(_figure_path(title), bbox_inches = 'tight')
    plt.close()


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
