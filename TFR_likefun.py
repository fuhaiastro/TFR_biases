""" 
Likelihood Functions for Forward, Inverse, and Dual-Scatter Models
"""
import numpy as np
from scipy.special import erfc

""" Probability and Likelihood Functions for the Dual-Scatter Model """
def prob_uTFR(bi,fifft, md,mld,bw,sigm, bsigw,VF_a,VF_vs):
    """ conditional pdf of m: p(m|w,d ; sig_m, sig_w, beta, alpha)"""
    # integral over i -> (f*g)(w)
    lnVF = 2.3025851*(VF_a+1)*(bi-VF_vs) - 10**(bi-VF_vs)
    gexp = np.exp(-(md-bi)**2/(2*sigm**2) + lnVF) 
    gerf = erfc((mld-bi)/(1.4142135*sigm)) * np.exp(lnVF) 
    # convolution by FFT then adjust to center
    Fexp = np.fft.ifft(np.fft.fft(gexp)*fifft).real
    Ferf = np.fft.ifft(np.fft.fft(gerf)*fifft).real
    mid = len(gexp) // 2 - 1
    Fexp = np.concatenate((Fexp[mid:], Fexp[:mid]))
    Ferf = np.concatenate((Ferf[mid:], Ferf[:mid]))
    # integral over w -> (F*G)(\tilde{w})
    Gfun = np.exp(-(bw-bi)**2/(2*bsigw**2))
    num = np.trapz(Fexp*Gfun)
    den = np.trapz(Ferf*Gfun)
    # get the ratio
    if num > 0 and den > 0:
        const = np.sqrt(2/np.pi)/sigm # the term before the integrals
        return const*num/den
    return 0.0

def ln_like_uTFR(theta, bounds, ws, ms, ds, ml):
    """ ln likelihood of the data set {ws, ms, ds} 
    given the model (theta), selection function (ml) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # get model parameters (c = gamma, b = beta, VF_vs = v_\star, VF_a = \alpha)
        c, b, sigm, sigw, VF_vs, VF_a = theta
        # hat_i = log sin i
        hi = np.linspace(-2.5,2.5,1024) 
        # pdf of hat_i for random orientation
        msk = hi < 0 
        fi = np.zeros_like(hi)
        fi[msk] = 10**(2*hi[msk]) / np.sqrt(1 - 10**(2*hi[msk]))
        # Fourier transform of p(i)
        fifft = np.fft.fft(fi)
        # bi = beta * log sin i
        bi = b*hi
        bsigw = b*sigw
        # input data arrays
        mds  = ms+ds-c
        mlds = ml+ds-c
        bws  = b*ws
        # compute prob for every data point
        ps = np.array([prob_uTFR(bi,fifft, md,mld,bw, sigm,bsigw,VF_a,VF_vs) for (md,mld,bw) in zip(mds,mlds,bws)])
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Probability and Likelihood Functions for the Forward Model """
def prob_dTFR(bi,ft, x,xl,bw, sigm,alpha):
    """ conditional pdf of m: p(m|w,d ; sig_m, beta, alpha)"""
    # velocity function, (blogW - bv*) - blogsini = b log(W/sini) - bv*
    lnVF = 2.3025851*(alpha+1)*(bw-bi) - 10**(bw-bi)
    # x+blogsini = logM-(blogW+c) + blogsini = logM-(blog(W/sini)+c)
    fexp = np.exp(-(x+bi)**2/(2*sigm**2)+lnVF) * ft
    ferf = erfc((xl+bi)/(1.4142135*sigm))*np.exp(lnVF) * ft
    # simple integration
    num = np.trapz(fexp)
    den = np.trapz(ferf) 
    if num > 0 and den > 0:
        const = np.sqrt(2/np.pi)/sigm # the term before the integrals
        return const*num/den
    return 0.0

def ln_like_dTFR(theta, bounds, ws, ms, ds, ml):
    """ ln likelihood of the data set {ws, ms, ds} 
    given the model (theta), selection function (ml) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # intercept, slope, m dispersion, velocity noise,
        # velocity function bv*, and faint-end slope
        c, b, sigm, VF_vs, VF_a = theta
        # t = sini^2 from 0 to 1
        t = np.linspace(1e-4, 0.9999, 1000)
        ft = 1/np.sqrt(1-t)
        bi = b * np.log10(t)/2
        # input data arrays
        xs  = ms+ds-(b*ws+c) # deviate between observed and TFR mass
        xls = ml+ds-(b*ws+c) # deviate between mass limit and TFR mass
        bws = b*ws-VF_vs     # bw - v* = logX for Schechter velocity function
        # compute conditional prob for each data point
        ps = np.array([prob_dTFR(bi,ft, x,xl,bw, sigm,VF_a) for (x,xl,bw) in zip(xs,xls,bws)])
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf

""" Likelihood Function for the Inverse Model """
def ln_like_iTFR(theta, bounds, ws, ms, ds):
    """ ln likelihood of the data set {ws, ms, ds} given the model (theta) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # BTFR intercept, slope, dispersion in logW
        c, b, sigw = theta
        # t := sini^2 from 0 to 1
        t = np.linspace(1e-4, 0.9999, 1000)
        hi = np.log10(t)/2 # hat_i := logsini = (log sini^2)/2
        ft = 1/np.sqrt(1-t) # pdf(t)
        dt = t[1]-t[0]
        # deviates between TFR-predicted and observed logW 
        xs = (ms+ds-c)/b - ws
        # compute prob for each data point
        ps = np.array([np.trapz(np.exp(-(x+hi)**2/(2*sigw**2)) * ft) for x in xs])
        # properly normalize
        ps *= dt/(2*np.sqrt(2*np.pi)*sigw)
        # return the sum of the logarithmic
        return np.sum(np.log(ps[ps > 0]))
    return -np.inf