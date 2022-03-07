'''
Functions used for generating and organising the variational time-stepping scheme used in solving the double-glazing problem.
'''

import numpy as np

# code a function to compute the next step
def next_dt(dt_min,
            dt_max,
            prev_t,
            tau,
            reg
            ):
    '''
    Calculate the next dt according to variation time-stepping scheme.

    Inputs:
        dt_min:     Float:      Initial dt value.
        dt_max:     Float:      Limit of dt values.
        prev_t:     Float:      Time of the previous step.
        tau:        Float:      Rate of growth of hot wall.
        reg:        Float:      Regularisation constant.    
    Outputs:
        dt:         Float:      The next dt in the sequence.
    '''
    dt = (dt_max-dt_min)*(1-np.exp(-prev_t/tau*1/reg))+dt_min
    return dt

# # Code a function to yield a list of the time steps at each time
# def gen_dtlist(dtmin,
#                tau,
#                T,
#                reg):
#     dtmax = 0.05 #10*dtmin #the max time steps is 10 times the smaller time steps 
#     t=0 #time = 0 to start
#     dt_list = [] #create an empty list to collect the values of the steps 
#     t_list = [] #create an empty list to collect at which time we measure the time steps 

#     while t<T+dtmin: #loop until we arrive at time t

#         dtnext = np.round(next_dt(dtmin,dtmax,t,tau,reg),4) #compute the next dt
#         dt_list.append(dtnext) #append it to the list of dt     
#         t+=dtnext #compute the next time 
#         t = round(t,10) #avoid numerical error coming from numerical scheme
#         t_list.append(t) #append it to the list of dt 

#     return t_list, dt_list #return the list of time and dt

def find_t_list(dt_min,
                dt_max,
                tau,
                reg,
                num_steps
                ):
    '''
    Generate list of times at which snapshots of solutions are evaluated given paramters and number of steps.

    Inputs:
        dt_min:     Float:      Initial dt value.
        dt_max:     Float:      Limit of dt values.
        tau:        Float:      Rate of growth of hot wall.
        reg:        Float:      Regularisation constant.
        num_steps:  Int:        Number of steps taken in time.
    Outputs:
        t_list:     List:       List of times.
    '''
    t = 0
    t_list = [0]
    
    for _ in range(num_steps):

        dt = next_dt(dt_min, dt_max, t, tau, reg)
        t += dt
        t = round(t, 10)
        t_list.append(t.copy())
    
    return t_list

def time_organiser(y_data,
                   dt_min,
                   dt_max,
                   tau,
                   reg,
                   num_steps,
                   data_res=0.00001
                   ):
    '''
    Arranges y_data snapshots evaluated with variable time-stepping to match likelihood data in time.

    Inputs:
        y_data:     np.Array:   Data generated using high resolution constant time-stepping solver.
        dt_min:     Float:      Initial dt value.
        dt_max:     Float:      Limit of dt values.
        tau:        Float:      Rate of growth of hot wall.
        reg:        Float:      Regularisation constant.
        num_steps:  Int:        Number of steps taken in time.
        data_res:   Float:      Resolution of fine data.
    Outputs:
        y_coarse:   np.Array:   Highly resolved data thinned to coincide with vairbale time-stepping scheme.
    '''
    t_list = find_t_list(dt_min,
                         dt_max,
                         tau,
                         reg,
                         num_steps
                         )

    y_coarse = []
    
    for t in t_list:
        y_coarse.append(y_data[int(t/data_res)])
        
    return y_coarse