import fenics
import numpy as np
import matplotlib.pyplot as plt
import typing
from typing import Callable, Tuple, List, Union
import pickle
from solver_alix import solver_para, solver_run
import statsmodels.api as sm


name_file = "Alixcode/alpha_res/chain-M=1000.csv"
# with open(name_file, 'wb') as f:
#     data = pickle.load(f)

with open(name_file, 'rb') as f:
    x = pickle.load(f)

x = x[100::]
# # Adding plot title.
plt.title("Autocorrelation Plot")
 
# Providing x-axis name.
plt.xlabel("Lags")
 
 
# Plotting the Autocorrelation plot.
plt.acorr(x, maxlags = 50)
 
# Displaying the plot.
print("The Autocorrelation plot for the data is:")
plt.grid(True)
 
plt.show()

## Compute Gelman Rubin stat

print(sm.tsa.acf(x))
