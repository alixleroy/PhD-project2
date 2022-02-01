## 0 - Import packages 
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
from SmoothBondary_time_double_galzing import time_double_glazing_smooth
#import scipy.stats as stats 

## 1 - Generate the observed data 
alpha = 0 #alpha star 
num_steps0 = 5
nx = ny = 5
u_mn = time_double_glazing_smooth(tau=1/10,
                    epsilon = 1/200,
                    w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=1),
                    num_steps = num_steps0,
                    T = 1.0,
                    nx = 5,
                    ny = 5,
                    k = 1
                    )

# compute noise and observed data
u_mn = np.array(u_mn)

noise  = np.random.normal(0,1,np.shape(u_mn))
print(noise)