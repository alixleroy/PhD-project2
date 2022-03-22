import numpy as np
from Functions_Library import compare_diagnosis_plots, load_data


chain_data = [(np.load('chains/var_metropolis.npy'), load_data('chains/alpha_list_RWMH')),
            (np.load('chains/var_delayed_MCMC.npy'), load_data('chains/alpha_list_DA')),
            (np.load('chains/var_gamma_nonreversible_MCMC.npy'), load_data('chains/alpha_list_IRWMH'))]

acceptance_ratio_data = [(np.load('chains/var_metropolis_accept_ratio.npy'), load_data('chains/A_list_RWMH')),
                        (np.load('chains/var_delayed_MCMC_accept_ratio.npy'), load_data('chains/A_list_DA')),
                        (np.load('chains/var_gamma_nonreversible_MCMC_accept_ratio.npy'), load_data('chains/A_list_IRWMH'))]

labels = [('RWMH: Cell averaging', 'RWMH: Interpolation'), ('Delayed acceptance: Cell averaging', 'Delayed acceptance: Interpolation'),
          ('Irreversible RWMH: Cell averaging', 'Irreversible RWMH: Interpolation')]

fnames = ['RWMH', 'DA', 'IRWMH']

for chains, acc_ratio, label, fname in zip(chain_data, acceptance_ratio_data, labels, fnames):
    compare_diagnosis_plots(mcmc_chains=chains,
                            mcmc_acceptance_ratios=acc_ratio,
                            lag=20,
                            nbins=50,
                            labels=label,
                            file_name=fname)
