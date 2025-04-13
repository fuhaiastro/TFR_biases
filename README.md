# Unbiased Regression of the Tully-Fisher Relation without Galaxy Inclinations 

*H. Fu, April 2025*

This repository provides the Python functions that implemented the Bayesian methods described in *Mitigating Malmquist and Eddington Biases in Latent-Inclination Regression of the Tully-Fisher Relation* (Hai Fu, 2025, Astrophysical Journal, submitted). The PDF file of the paper is provided (`TFR_biases.pdf`). 

The Python notebook `simu2fit.ipynb` gives examples on how to simulate galaxy samples, how to run the models, and how to illustrate the posterior pdfs from the MCMC sampler. 

As a note on the performance of the code, below I list the computing time for 1,000 MCMC steps on a 2021 M1 Pro Macbook Pro with 8 performance CPU cores, 2 efficiency CPU cores, and 16 GPU cores. The size of input galaxy sample is 10,147 for all models.

- Forward model, 16 walkers
    - `100%|██████████| 1000/1000 [14:24<00:00,  1.16it/s]`
- Inverse model, 8 walkers
    - `100%|██████████| 1000/1000 [03:36<00:00,  4.63it/s]`
- Dual-Scatter model (on CPU), 16 walkers
    - `100%|██████████| 1000/1000 [27:25<00:00,  1.65s/it]`
- Dual-Scatter model (on GPU), 16 walkers
    - `100%|██████████| 1000/1000 [07:19<00:00,  2.28it/s]`