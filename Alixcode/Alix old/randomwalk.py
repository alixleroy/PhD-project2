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

def log_ratio(y: np.array, u1: np.array, u2:np.array, alpha1: float, alpha2: float, sigma_p: float, sigma_l: float, mu_p: float) -> np.array:
    '''
    
    '''
    return 0.5 * ((((alpha1 - mu_p) ** 2 - (alpha2 - mu_p) ** 2) / sigma_p ** 2) + (np.linalg.norm(y - u1) ** 2 - np.linalg.norm(y - u2) ** 2) / sigma_l ** 2)


### b - Random Walk Algorithm 

#### lenght of chain 
M = 50
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

# # list of saved values for alpha 
alpha_list = np.zeros((M,1))

# save all proposed alpha 
alpha_proposed = []

# save all logratio 
logratiosL = []
for i in range(M):
    # get a guess: alpha2 
    alpha2 = np.random.normal(alpha1,sigmaG,1)
    alpha_proposed.append(alpha2)

    #get u2 
    u2=solver_dg(wind(alpha2),num_steps_true, dt_true,tau, epsilon,mesh_solver,V_solver,u_D,bc,u_n,u,v,f) # run the solver accross time

    #compute ratio from alpha1 to proposal alpha 2 
    logratio12 = log_ratio(y_sols,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP)
    logratiosL.append(logratio12)

    # print("Log ratio"+str(logratio12))
    #draw h from uniform(0,1)
    h = np.log(np.random.uniform(0,1))
    # print("h = "+str(h))
    if h <= np.min([0,logratio12]):
        alpha1=alpha2
        u1=u2

    alpha_list[i] = alpha1

        

#plot the values in the chain
print("Proposed alpha")
print(alpha_proposed)

print("\\")
print("Logratios")
print(logratiosL)

print("\\")
print("Alpha list")
print(alpha_list)

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(18, 5)) #create the figure 
ax1.plot(alpha_list) #plot values of the element 0 of the chain 

bins = np.linspace(-3., 3., 100) #set bins to plot an histogram of the values of X^0
step = bins[1]-bins[0] #adjust the setp 
hist,bin_edges = np.histogram(alpha_list, bins=bins,density=True) # get the histogram
hist_norm = hist/(sum(hist)*step) #normalise the histogram
ax2.bar(bin_edges[:-1], hist_norm, width = step, alpha = 0.5) #plot the histogram

fig.savefig("Alixcode/markov_chain.png")

name_file = "Alixcode/alpha_res/chain-M="+str(M)+".csv"
with open(name_file, 'wb') as f:
    pickle.dump(alpha_list, f)

# ax1.set_title("Values of $\alpha$") #title to the plot 
# ax1.set_xlabel("Iteration") #set label
# ax1.set_ylabel("$\alpha$") #set label 
# nstart=int(0) #set burn in period 
# bins = np.linspace(-3., 3., 100) #set bins to plot an histogram of the values of X^0
# step = bins[1]-bins[0] #adjust the setp 
# # xnormal = stats.norm.pdf(bins, 0, 1) #draw a normal centered gaussian to compare to the results 
# hist,bin_edges = np.histogram(alpha_list[0,nstart::], bins=bins,density=True) # get the histogram
# hist_norm = hist/(sum(hist)*step) #normalise the histogram
# ax2.bar(bin_edges[:-1], hist_norm, width = step, alpha = 0.5) #plot the histogram

# # xnormal = stats.norm.pdf(bins, 0, 1) #draw a normal centered gaussian to compare to the results 
# # ax2.plot(bins,xnormal) #plot the centered normal gaussian
# ax2.set_xlim(min(bin_edges), max(bin_edges)) #set limit to the plot axis
# ax2.grid(axis='y', alpha=0.75) #set a grid 
# ax2.set_xlabel('Value',fontsize=10) #set a label to axis x
# ax2.set_ylabel("Distribution of the values of X0",fontsize=10) #set a label to axis y
# ax2.set_title("Histograms of the markov chain for $X^{(0)}$")



