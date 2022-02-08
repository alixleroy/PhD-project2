

from solver_rasheed import pred_exponential_time_dg,fenics_essentials_exponential,exponential_time_double_glazing
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from vedo.dolfin import plot
import warnings
import logging

u_rasheed = exponential_time_double_glazing(0,
                                    1/10,
                                    epsilon = 1/200,
                                    w = Expression(('2*x[1]*(1-x[0]*x[0])', '-2*x[0]*(1-x[1]*x[1])'), degree=3),
                                    num_steps = 50,
                                    T = 5.0,
                                    nx = 10,
                                    ny = 10,
                                    k = 1,
                                    plot_u=True,
                                    plot_u_t=True)