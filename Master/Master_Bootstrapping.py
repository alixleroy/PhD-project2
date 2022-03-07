'''
Implementation of weighted likelihood bootstrapping algorithm for the double-glazing inverse problem. Referece paper: "Introducing prior information in Weighted Likelihood Bootstrap
with applications to model misspecification" by Emily Pompe.
'''
import numpy as np

def log_likelihood(alpha,
                   sigma_l
                   ):
def bootstrapping(w_0,
                  n,
                  iterations,
                  ):
    '''
    Performs iterations of the weighted likelihood bootstrapping algorithm to generate a sampling distribution for alpha in the double-glazing
    inverse problem.
    '''

    for j in iterations:
        
        w = np.random.exponential(1, n)

        sum
        for i in range(n):


        alpha_j = np.argmax()