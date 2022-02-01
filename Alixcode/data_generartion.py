
## 0 - Import the function required to generate the data 
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
from SmoothBondary_time_double_galzing import time_double_glazing_smooth

## I - Smooth boundary condition with a unperturbed wind

# u = time_double_glazing_smooth(tau=1/10,
#                     epsilon = 1/200,
#                     w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
#                     num_steps = 5,
#                     T = 5.0,
#                     nx = 5,
#                     ny = 5,
#                     k = 1
#                     )

# print(np.shape(u))

# print(u)

## II - Smooth boundary condition with a perturbed wind with multiplicative coefficient 
alpha = 0 #alpha star 
num_steps0 = 10
nx = ny = 5
u_mn = time_double_glazing_smooth(tau=1/10,
                    epsilon = 1/200,
                    w = Expression(('alpha*2*x[1]*(1-x[0]*x[0])', 'alpha*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=1),
                    num_steps = num_steps0,
                    T = 1.0,
                    nx = 5,
                    ny = 5,
                    k = 1
                    )
# compute noise 
u_mn = np.array(u_mn)
print(np.shape(u_mn))

noise  = np.random.random(np.shape(u_mn))

y = u_mn + noise 

## Save the data u
smooth_boundary_winda = "smoothboundary_data/u_smooth_boundary_winda-ns="+str(num_steps0)+"-a="+str(alpha)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(u_mn, f)

## Save the observed data y
smooth_boundary_winda = "smoothboundary_data/y_smooth_boundary_winda-ns="+str(num_steps0)+"-a="+str(alpha)+".csv"
with open(smooth_boundary_winda, 'wb') as f:
    pickle.dump(y, f)

print(np.shape(y))
