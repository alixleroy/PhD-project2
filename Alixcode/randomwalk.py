## 0 - Import packages 
from fenics import *
import numpy as np
from vedo.dolfin import plot, Latex, clear, histogram
import matplotlib.pyplot as plt
import pickle 
from SmoothBondary_time_double_galzing import time_double_glazing_smooth
#import scipy.stats as stats 

## 1 - Generate the observed data 
alpha_star = 0 #alpha star 
num_steps0 = 5
nx = ny = 5
u_mn = time_double_glazing_smooth(tau=1/10,
                    epsilon = 1/200,
                    w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=alpha_star),
                    num_steps = num_steps0,
                    T = 1.0,
                    nx = 5,
                    ny = 5,
                    k = 1
                    )

# compute noise and observed data
u_mn = np.array(u_mn)
noise  = np.random.normal(0,1,np.shape(u_mn))

true_y = u_mn + noise 

## 2 - Metropolis hasting 

### a- compute ratio 
def ratio(y,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP):
    ratio = np.exp(1/(2*sigmaL**2)*(np.linalg.norm(y-u1)**2-np.linalg.norm(y-u2)**2)+ 1/(2*sigmaP**2)*(np.linalg.norm(alpha1-muP)**2-np.linalg.norm(alpha2-muP)**2)  ) 
    return ratio

def logratio(y,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP):
    ratio = 1/(2*sigmaL**2)*(np.linalg.norm(y-u1)**2-np.linalg.norm(y-u2)**2)+ 1/(2*sigmaP**2)*(np.linalg.norm(alpha1-muP)**2-np.linalg.norm(alpha2-muP)**2)
    return ratio

### b - Random Walk Algorithm 

#### lenght of chain 
M = 10000

#### Decide on prior distribution 
muP = 0
sigmaP = 0.5

#### Decide on likelihood distribution 
sigmaL = 0.5

#### Run the loop 
#### intial choice for alpha 
alpha1 = np.random.normal(0,1,1)


# get u1 
u1 = time_double_glazing_smooth(tau=1/10,
                    epsilon = 1/200,
                    w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=alpha1),
                    num_steps = num_steps0,
                    T = 1.0,
                    nx = nx,
                    ny = nx,
                    k = 1
)

# list of saved values for alpha 
alpha_list = np.zeros((M,1))

for i in range(M):


    # get a guess: alpha2 
    alpha2 = np.random.normal(alpha1,0.1,1)

    #get u2 
    u2 = time_double_glazing_smooth(tau=1/10,
                epsilon = 1/200,
                w = Expression(('exp(alpha)*2*x[1]*(1-x[0]*x[0])', 'exp(alpha)*-2*x[0]*(1-x[1]*x[1])'), degree=3,alpha=alpha2),
                num_steps = num_steps0,
                T = 1.0,
                nx = nx,
                ny = ny,
                k = 1
                )
    #compute ratio from alpha1 to proposal alpha 2 
    logratio12 = logratio(true_y,u1,u2,alpha1,alpha2,sigmaP,sigmaL,muP)
        

    #draw h from uniform(0,1)
    h = np.log(np.random.uniform(0,1))
    # print(h)
    # print(logratio12)
    if h <= np.min([0,logratio12]):
        alpha1=alpha2
        u1=u2

    alpha_list[i] = alpha1

        

#plot the values in the chain

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



