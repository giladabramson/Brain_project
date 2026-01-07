# Script that, given an Hopfield Model, implements the (stochastic)
# Euler-Mayorama method of integration for the solution of a given SDE
# time [0,T]

# Author: Simone Betteti (2023)
# Paper: Stimulus-Driven Dynamics for Robust Memory Retrieval in Hopfield Networks

import numpy as np
from tqdm import tqdm

class EM:
    # Initialization of the integrator
    def __init__(self, HN, t_ini, t_end, dt, C, I, w_external=1.0, w_s=1.0, stim_duration=1.0):
        # definition of the initial integration time
        self.t0 = t_ini
        # definition of the final integration time
        self.T = t_end
        # definition of the step size
        self.dt = dt
        # slope of the activation function
        self.delta = 10
        # Tolerance for the equilibrium condition
        self.tol = 10**(-9)
        # Definition of the timescales for the two layers
        self.tau1 = 1
        self.tau2 = 1
        # Integration condition
        # C == 'M' --> Multiplicative
        # C == 'A' --> Additive
        self.C = C
        # Input condition
        # I == 'm' --> modulated
        # I == 'c' --> constant
        self.In = I
        # Scaling factors for external input and recurrent term
        self.w_external = w_external
        self.w_s = w_s
        # Duration (in model seconds) the stimulus is actively applied
        self.stim_duration = stim_duration
    
    # Function that implements the Euler Mayorama method given an SDE
    def Eu_Ma_Test(self,HN, sigma, u):

        # Definition of the integration interval, appropriately partitioned
        x = np.arange(start=self.t0, stop=self.T, step=self.dt)
        # defintion of the solution vector
        y = np.zeros((HN.N,np.size(x)))
        # Generation of a binary mask for possible dilution of the synaptic matrix
        M = np.random.choice([0,1], size=(HN.N,HN.N), p=[0, 1])

        # setting of the initial condition
        y[:,0] = HN.y0
        # Solution of the system using Euler-Mayorama method
        for t in tqdm(range(np.size(x)-1)):
            if self.C == 'M':
                try:
                    I = u[:,t]
                except:
                    I = u
            elif self.C == 'A':
                if self.In == 'm':
                    I = np.zeros((HN.N,))
                    stim_end = self.stim_duration if self.stim_duration is not None else 1.0
                    try:
                        if x[t] < stim_end:
                            I = u[:,t]
                    except:
                        if x[t] < stim_end:
                            I = u
                else:
                    try:
                        I = u[:,t]
                    except:
                        I = u
            # Computation of the value field
            Hf = self.hop_field_test(y[:,t],HN, I, M)
            # Computation of t:he buffer value field
            buff_y = y[:,t] + Hf*(self.dt+sigma*np.sqrt(self.dt))    # Needed if the noise is state-dependent
            # Associated filtered quantities
            gHf = np.ones((HN.N,))
            gHf_buff = np.ones((HN.N,))
            # Computation of the Brownian increments
            dW = np.random.randn(np.shape(y[:,t])[0],)
            # Computation of the field updated value
            y[:,t+1] = y[:,t] + (Hf*self.dt + np.sqrt(self.dt)*np.multiply(sigma*gHf,dW) + (1/(np.sqrt(self.dt)*2))*np.multiply(sigma*(gHf_buff-gHf),(dW**(2)-self.dt*np.ones((np.shape(y[:,t])[0],)))))

        return y


    # Function that implements the activation function for a given vector of currents
    def g_fun_test(self,y):

        z = np.tanh(self.delta*y)

        return z
    
    # Function that implements the computation of the Hopfield-field (hehe)
    def hop_field_test(self, y, HN, u, M):
        # Filtered activation
        z = self.g_fun_test(y)
        # Construction of the interaction matrix
        B = range(HN.N)
        if self.C == 'M':   
            Dalph = (1/HN.N)*np.diagflat(np.dot(np.transpose(HN.mems),u))
            W = (1/HN.N)*np.multiply(M,HN.mems@Dalph@np.transpose(HN.mems))
            W[B,B] = 0
            # Value of the field
            val = -y + self.w_s*np.dot(W,z)
        elif self.C == 'A': #
            W = (1/HN.N)*np.multiply(M,HN.mems@np.transpose(HN.mems))
            W[B,B] = 0
            # Value of the field
            val = -y + self.w_s*np.dot(W,z) + self.w_external*u #add factor to multiply u an np.dot(W,z), where u is the input.

        return val
