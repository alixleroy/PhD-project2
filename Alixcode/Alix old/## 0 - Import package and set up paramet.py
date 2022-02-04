## 0 - Import package and set up parameters

from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
from solver_functions import solver_parameters, solver_dg

## I - Set up the parameters of the PDE problems to resolve 
T = 1 #until T =
tau = 5 #value of tau
epsilon=1/200 #value of epsilon
def wind(alpha): #define the value of the wind 
    return Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=alpha)

## II- Solve the problem and obtain observed data
num_steps_true = 100 #number of steps to take

nx_true=ny_true = 20 #space mesh
dt_true = T / num_steps_true # time step size
mesh_solver,V_solver,u_D,bc,u_n,u,v,f = solver_parameters(nx_true,ny_true,tau,epsilon) #run the background info needed to solve pde
u_sols = solver_dg(wind(0),num_steps_true, dt_true,tau, epsilon,mesh_solver,V_solver,u_D,bc,u_n,u,v,f) #run the solver accross time
y_sols = u_sols+ np.random.normal(0,1,np.shape(u_sols)) #get the true solutions by adding noise 

## III- Metropolis hasting 

### a- compute ratio 
def ratio(y,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP):
    ratio = np.exp(1/(2*sigmaL**2)*(np.linalg.norm(y-u1)**2-np.linalg.norm(y-u2)**2)+ 1/(2*sigmaP**2)*(np.linalg.norm(alpha1-muP)**2-np.linalg.norm(alpha2-muP)**2)  ) 
    return ratio

def logratio(y,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP):
    ratio = 1/(2*sigmaL**2)*(np.linalg.norm(y-u1)**2-np.linalg.norm(y-u2)**2)+ 1/(2*sigmaP**2)*(np.linalg.norm(alpha1-muP)**2-np.linalg.norm(alpha2-muP)**2)
    return ratio

### b - Random Walk Algorithm 

#### lenght of chain 
M = 10
sigmaG = 0.5 #1/M #variance of the proposal distribution 

### Decide on prior distribution 
muP = 0
sigmaP = 1

#### Decide on likelihood distribution 
sigmaL = 1

#### Run the loop 
#### intial choice for alpha 
alpha1 = np.random.normal(0,1,1)

#### intial parameters for a PDE solver 
nx=ny=nx_true #for the PDE to solve in MCMC typically decrease the number of spatial steps taken
num_steps_true = num_steps_true #for the PDE to solve in MCMC typically decrease the number of time steps taken
dt_true = T/num_steps_true #typically get the dt
mesh_solver,V_solver,u_D,bc,u_n,u,v,f = solver_parameters(nx,ny,tau,epsilon) # run the background info needed to solve pde
u1 = solver_dg(wind(alpha1),num_steps_true, dt_true,tau, epsilon,mesh_solver,V_solver,u_D,bc,u_n,u,v,f) # run the solver accross time
