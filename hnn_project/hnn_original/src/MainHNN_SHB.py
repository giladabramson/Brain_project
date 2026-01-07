# Script for the implementation of an input driven Hopfield model with 
# either only short term synaptic plasticity or both short and long
# term synaptic components

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from HNN_Gen import HNN
from Eul_May import EM
from HNPlot import PlotOverlap, PlotOverlapCon

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
# Defines the scalar amplitude of the perturbations
sigma = [0.5]
#sigma = [0, 2, 4]
d_sigma = len(sigma)

# Definition of the integration interval
t_ini = 0
t_end = 10
# Definition of the time step
dt = 0.01

# Generation of the single input window
x = np.arange(start=0,stop=t_end,step=dt)
T = np.size(x)

# Generation of the Hopfield model
HN = HNN(C,eps)
# Generation of the memories
HN.net()
# Generation of the initial condition
HN.y0_gen()
# Storing of the original value 
y_buff = HN.y0

# Define the length of the input sequence
n = 2
# Define the number of samples
S = 50

# Definition of the inputs
u_proto = np.random.uniform(low=0.8,high=1.5,size=(HN.P, n, S))
# u_proto = np.random.uniform(low=2,high=5,size=(HN.P, n, S))
for s in range(S):
    for j in range(n):
        u_proto[j,j,s] = np.random.uniform(low=2,high=3.5)
        # u_proto[j,j,s] = np.random.uniform(low=7,high=10)
        if j>0:
             u_proto[j-1,j,s] = np.random.uniform(low=0.05,high=0.2)

# Definition of the colormap for the weights
col = ['cyan' for k in range(HN.P)]
col[0] = 'red'
col[1] = 'green'
col[2] = 'blue'
col[3] = 'black'

u_tran = np.zeros((HN.N,n,S))
for s in range(S):
    for h in range(n):
        for l in range(HN.P):
            #u[:,h] += u_proto[l,h]*HN.mems[:,l]
            u_tran[:, h, s] += u_proto[l, h, s]*HN.mems[:,l]

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
            # EMa = EM(HN, t_ini, t_end, dt, 'M','c')                 # SDP Hopfield
            EMa_add = EM(HN, t_ini, t_end, dt, 'A','m')             # Classic Hopfield - Modulated Stimulus
            #EMa_add_c = EM(HN, t_ini, t_end, dt, 'A','c')           # Classic Hopfield - Constant Stimulus

            # Integration of the system over the time interval
            # IDP Hopfield - Uncomment if needed
            # if j>0:
            #     HN.y0 = y[:,-1]
            # y = EMa.Eu_Ma_Test(HN, sigma[d], u_tran[:,j,s])

            # Classic Hopfield w modulated input - Uncomment if needed
            if j>0:
               HN.y0 = y_add[:,-1]
            y_add = EMa_add.Eu_Ma_Test(HN, sigma[d], u_tran[:,j,s])

            # Classic Hopfield w constant input - Uncomment if needed
            #if j>0:
            #    HN.y0 = y_add_c[:,-1]
            #y_add_c = EMa_add_c.Eu_Ma_Test(HN, sigma, u_tran[:,j,s])

            # Update of the trajectories
            # Y[:,j*T:(j+1)*T, s, d] = y
            Y_add[:,j*T:(j+1)*T, s, d] = y_add
            #Y_add_c[:,j*T:(j+1)*T, s,d] = y_add_c

# Plotting of the overlap during training
if d_sigma == 1:
    # PlotOverlap(HN, EMa, Y, n*t_end, col, n, sigma, 'c')
    PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, sigma, 'm')
    #PlotOverlap(HN, EMa_add_c, Y_add_c, n*t_end, col, n, sigma, 'c')
else:
    PlotOverlapCon(HN, Y, n*t_end, d_sigma, n)


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
#         EMa = EM(HN, t_ini, t_end, dt, 'M', 'c')
#         EMa_add = EM(HN, t_ini, t_end, dt, 'A', 'm')
#         # Integration of the system over the time interval
#         y = EMa.Eu_Ma_Test(HN, sigma, u_tot[:,:, s])
#         y_add = EMa_add.Eu_Ma_Test(HN, sigma, u_tot[:,:,s])

#     Y[:,j*T:(j+1)*T, s] = y
#     Y_add[:,j*T:(j+1)*T, s] = y_add


# # Plotting of the overlap during training
# PlotOverlap(HN, EMa, Y, n*t_end, col, n, sigma, 'c')
# PlotOverlap(HN, EMa_add, Y_add, n*t_end, col, n, sigma, 'm')