import fenics
import numpy as np
import matplotlib.pyplot as plt
import typing
from typing import Callable, Tuple, List, Union
import pickle
from solver_alix import solver_para, solver_run
import statsmodels.api as sm
#from statsmodels.graphics.tsaplots import plot_acf

## I - Import the autocorrelations 
name_file = "Alixcode/alpha_res/chain-M=5000.csv"
with open(name_file, 'rb') as f:
    x1 = pickle.load(f)

name_file = "Alixcode/alpha_res/chain-M=1000.csv"
with open(name_file, 'rb') as f:
    x2 = pickle.load(f)

## II - Burn in periods 
x1 = x1[100::]
x2 = x2[100::]

## III - Design autocorrelation function 

def auto_corr_fast(M,kappa):   
#   The autocorrelation has to be truncated at some point so there are enough
#   data points constructing each lag. Let kappa be the cutoff
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)
#   G is the autocorrelation curve
    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G

autocorr1 = auto_corr_fast(x1,50)
autocorr2 = auto_corr_fast(x2,50)
plt.plot(autocorr1,label="M=5000")
plt.plot(autocorr2,label="M=1000")
plt.legend()
plt.show()

## IV Autocorrelation function
# plt.title("Autocorrelation Plot")
 
# # Providing x-axis name.
# plt.xlabel("Lags")

# # Plotting the Autocorrelation plot.
# plt.acorr(x1, maxlags = 50)
# plt.acorr(x2, maxlags = 50)

# # Displaying the plot.
# print("The Autocorrelation plot for the data is:")
# plt.grid(True)
 
# plt.show()

## Compute Gelman Rubin stat

#print(sm.tsa.acf(x))
