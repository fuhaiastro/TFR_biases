""" 
MCMC sampling of posterior with emcee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing, emcee, os, corner
from TFR_likefun import ln_like_dTFR, ln_like_iTFR, ln_like_uTFR

# prep for multithreading
multiprocessing.set_start_method("fork")   # fork: copy a Python process from an existing process.
os.environ["OMP_NUM_THREADS"] = "1"        # disable automatic parallelization in NumPy

def run_emcee(ws,ms,ds,                    # input data arrays: logW, logmb, d=2logD
            outdir='emcee',                # output folder
            method='forward',              # models: forward, inverse, or unified
            ml=5.736,                      # log apparent mass limit
            converge_check=False,          # check chain convergence before proceeding?
            ncpu=os.cpu_count(),           # number of CPUs to use
            nsteps=100, nrepeat=1):        # emcee iterations and number of repeats
    
    """ Validate Inputs """
    methods = {'forward','inverse','unified'}
    if method not in methods:
        raise ValueError(f"Invalid Input Parameters: {method=}")

    """ Output Folder """
    if not os.path.exists(outdir): os.mkdir(outdir)
    ndata = len(ds)
    suffix = f'd{np.median(ds):.2f}_n{ndata}'

    """ Model Parameters: latex names for plots and limits for flat priors """
    # sigma = 0 causes issues, so use 1e-3 as the minimum
    eps = 1e-3
    if method == 'forward':
        params = ["$\gamma$", "$\\beta$", "$\sigma_m$", "$\\beta v_*$", "$\\alpha$"]
        bounds = np.array([[10.0, 2.5, eps, -1.0, -2.0],
                           [11.0, 4.5, 0.3,  1.0,  0.0]])
    elif method == 'inverse':
        params = ["$\gamma$", "$\\beta$", "$\sigma_w$"]
        bounds = np.array([[10.0, 2.5, eps],
                           [11.0, 4.5, 0.1]])
    elif method == 'unified':
        params = ["$\gamma$", "$\\beta$", "$\sigma_m$", "$\sigma_w$", "$\\beta v_*$", "$\\alpha$"]
        bounds = np.array([[10.0, 2.5, eps, eps, -1.0, -2.0],
                           [11.0, 4.5, 0.3, 0.1,  1.0,  0.0]])

    """ initial starting positions of the walkers """
    ndim = len(bounds[0,:])
    nwalkers = ncpu*int(np.ceil(2.5*ndim/ncpu)) 
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
        # set up backend file to save MCMC samples
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

        with multiprocessing.Pool() as pool:
            if method == 'forward':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    ln_like_dTFR, args=(bounds,ws,ms,ds,ml), 
                    pool=pool, backend=backend)
            elif method == 'inverse':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    ln_like_iTFR, args=(bounds,ws,ms,ds), 
                    pool=pool, backend=backend)
            elif method == 'unified':
                sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                    ln_like_uTFR, args=(bounds,ws,ms,ds,ml), 
                    pool=pool, backend=backend)
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