""" 
MCMC sampling of posterior with emcee
(GPU version of the Unified Model)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee, os, corner
from TFR_likefun_gpu import ln_like_uTFR

def run_emcee_gpu(ws, ms, ds,              # input data: logW, logmb, 2logD
            outdir='emcee',                # output folder
            ml=5.736,                      # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            nwalkers=16,                   # number of walkers
            nsteps=100, nrepeat=1):        # emcee iterations and number of repeats

    """ Output Folder """
    method = 'unifgpu'
    ndata = len(ds)
    if not os.path.exists(outdir): os.mkdir(outdir)
    suffix = f'd{np.median(ds):.2f}_n{ndata}'

    """ Model Parameters: latex names for plots and limits for flat priors """
    # sigma = 0 causes issues, so use 1e-3 as the minimum
    eps = 1e-3
    params = ["$\gamma$", "$\\beta$", "$\sigma_m$", "$\sigma_w$", "$\\beta v_*$", "$\\alpha$"]
    bounds = np.array([[10.0, 2.5, eps, eps, -1.0, -2.0],
                       [11.0, 4.5, 0.3, 0.1,  1.0,  0.0]])

    """ initial starting positions of the walkers """
    ndim = len(bounds[0,:])
    cnter = (bounds[1,:]+bounds[0,:])/2 # distribution means 
    scale = (bounds[1,:]-bounds[0,:])/2 # width of uniform distribution
    pos = cnter + scale * (np.random.uniform(size=(nwalkers, ndim))-0.5)
    # clip initial positions outside of bounds
    for i in range(ndim):
        pos[:,i] = np.clip(pos[:,i], bounds[0,i], bounds[1,i]) 

    """ MCMC-sampling """
    print(f'\n{method=}, {ndata=}, {nsteps=}, {nrepeat=}, {nwalkers=}, {ndim=}')
    for irepeat in range(1,nrepeat+1):
        print(f'\nThis is {irepeat=} out of {nrepeat=}')
        # set up backend to save MCMC samples
        emcfile = outdir+f'/emcee_{suffix}.h5'
        backend = emcee.backends.HDFBackend(emcfile)
        if not os.path.exists(emcfile): 
            print(f'Reset backend because {emcfile} not present')
            backend.reset(nwalkers, ndim)
            print(f'Start MCMC from initial random positions')
            instate = pos
            chainexist = False
        else: 
            print('Start MCMC from where left off the last time')
            instate = None
            chainexist = True
        # set up sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                ln_like_uTFR, args=(bounds,ws,ms,ds,ml), 
                backend=backend)
        # check if chain is already long enough
        if chainexist and converge_check:
            chainshape = sampler.get_chain().shape
            tau = sampler.get_autocorr_time(quiet=True)
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))
            flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
            mids = np.array([np.percentile(flat_samples[:,i],50) for i in range(ndim)])
            if chainshape[0] > 50*np.max(tau): 
                print("[Break]: Chain length already exceeds 50x Auto-correlation Time")
                break                
        # start walking
        sampler.run_mcmc(instate, nsteps, progress=True)
        
        """ after nsteps, make a corner plot over full bounded range """
        # use trimmed, thinned, flattened sample for corner plots
        tau = sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        mids = np.array([np.percentile(flat_samples[:,i],50) for i in range(ndim)])
        _ = corner.corner(flat_samples, labels=params, 
                title_quantiles=[0.16,0.50,0.84], quantiles=[0.50],
                show_titles=True, plot_contours=False, plot_density=False, bins=50,
                range=list(zip(bounds[0],bounds[1])))
        plt.savefig(outdir+f'/corner_{suffix}_{(sampler.get_chain().shape)[0]}.png', bbox_inches='tight')
        plt.clf()

    """ return best-fit parameters and flattened chains """
    return mids, flat_samples