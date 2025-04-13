""" 
Likelihood Function for the Complete Model using PyTorch 
Mac's GPU device is called MPS (Metal Performance Shader)
"""
import torch
import numpy as np

# MPS prefers float32
dtype = np.float32  
# set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def prob_uTFR_vectorized(w, m, d, theta, ml):
    """
    Vectorized conditional pdf of m: p(m|w,d ; sig_m, sig_w, beta, alpha)
    Inputs:
        w, m, d: NumPy arrays of the same shape (e.g., (n,)), the input data
        theta: list, model parameters [c, b, sigm, sigw, VF_vs, VF_a]
        ml: scalar, the selection cutoff
    Returns:
        Tensor of shape (n,) with pdf values
    """
    theta = torch.tensor(theta, dtype=torch.float32, device=device)  # Convert theta to tensor
    w = torch.from_numpy(w.astype(dtype)).to(device)
    m = torch.from_numpy(m.astype(dtype)).to(device)
    d = torch.from_numpy(d.astype(dtype)).to(device)

    # Broadcast m, d, w to handle vectorized computation
    md = m + d - theta[0]  # Shape: (n,)
    mld = ml + d - theta[0]  # Shape: (n,)
    bw = w * theta[1]  # Shape: (n,)

    # Define i grid (same for all inputs, shape: (1024,))
    i = torch.linspace(-2.5, 2.5, 1024, device=device)
    bi = i * theta[1]  # Shape: (1024,)
    valid_mask = i < 0  # Shape: (1024,)
    fi = torch.zeros_like(i, device=device)
    fi[valid_mask] = 10**(2 * i[valid_mask]) / torch.sqrt(1 - 10**(2 * i[valid_mask]))

    # Velocity distribution function (shape: (1024,))
    lnVF = 2.3025851 * (theta[5] + 1) * (bi - theta[4]) - 10**(bi - theta[4])

    # Expand md and mld to (n, 1024) for broadcasting with bi
    md_expanded = md.unsqueeze(-1)  # Shape: (n, 1)
    mld_expanded = mld.unsqueeze(-1)  # Shape: (n, 1)
    bi_expanded = bi.unsqueeze(0)  # Shape: (1, 1024)

    # Compute gexp and gerf with broadcasting (shape: (n, 1024))
    gexp = torch.exp(-(md_expanded - bi_expanded)**2 / (2 * theta[2]**2) + lnVF)
    # Workaround for torch.erfc on MPS: use 1 - erf
    gerf = (1.0 - torch.erf((mld_expanded - bi_expanded) / (1.4142135 * theta[2]))) * torch.exp(lnVF)

    # FFT-based convolution (shape: (n, 1024))
    fifft = torch.fft.fft(fi)  # Shape: (1024,)
    Fexp = torch.fft.fftshift(torch.fft.ifft(torch.fft.fft(gexp, dim=1) * fifft, dim=1).real, dim=1)
    Ferf = torch.fft.fftshift(torch.fft.ifft(torch.fft.fft(gerf, dim=1) * fifft, dim=1).real, dim=1)

    # Expand bw to (n, 1024) for Gfun computation
    bw_expanded = bw.unsqueeze(-1)  # Shape: (n, 1)
    Gfun = torch.exp(-(bw_expanded - bi_expanded)**2 / (2 * (theta[1] * theta[3])**2))  # Shape: (n, 1024)

    # Sum over the i dimension (dim=1) to get num and den (shape: (n,))
    num = torch.sum(Fexp * Gfun, dim=1)
    den = torch.sum(Ferf * Gfun, dim=1)

    # Compute the ratio with masking for valid values
    const = 0.79788456 / theta[2]  # sqrt(2/PI)
    valid = (num > 0) & (den > 0)  # Shape: (n,)
    result = torch.zeros_like(num)  # Shape: (n,)
    result[valid] = const * num[valid] / den[valid]

    return result

def ln_like_uTFR(theta, bounds, ws, ms, ds, ml):
    """ ln likelihood of the data set {ws, ms, ds} 
    given the model (theta), selection function (ml), and velocity noise (sds) """
    # only calculate when pars are within bounds
    if np.all((theta > bounds[0, :]) & (theta < bounds[1, :])):
        # compute prob for every data point
        ps = prob_uTFR_vectorized(ws, ms, ds, theta=theta, ml=ml)
        # return the sum of the logarithmic to CPU as NumPy scaler
        return torch.sum(torch.log(ps[ps > 0])).cpu().numpy()
    return -np.inf