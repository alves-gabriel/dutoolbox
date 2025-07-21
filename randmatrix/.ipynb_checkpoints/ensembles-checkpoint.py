import numpy as np
import scipy as scp

###############
# CUE and GUE #
###############

def GUE_matrix(N, mu=0., sigma=1.):
    """Generates a random N x N matrix from the Gaussian unitary ensemble with with mean mu and variance sigma."""

    # Random complex normal
    GUE = np.random.normal(mu, sigma, size=(N, N)) + 1j*np.random.normal(mu, sigma, size=(N, N))
    
    # Symmetrizes
    return (GUE + GUE.conj().T)/np.sqrt(2)

def GOE_matrix(N, mu=0., sigma=1.):
    """Generates a random N x N matrix from the Gaussian orthogonal ensemble with with mean mu and variance sigma."""

    # Random real normal
    GOE = np.random.normal(mu, sigma, size=(N, N))

    return (GOE + GOE.conj().T)/np.sqrt(2)

def CUE_matrix(N):
    """Generates a random N x N matrix from the circular unitary ensemble"""
    return scp.stats.unitary_group.rvs(N)

#######################
# OTHER DISTRIBUTIONS #
#######################

def gamma_dist(k, lamb=1., size=None): 
    """ 
    See: https://github.com/scipy/scipy/blob/v1.13.0/scipy/stats/_continuous_distns.py#L3288-L3493
    for the scipy parametrization. 
    
    We use the wikipedia parametrization: https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    if size is None:
        return scp.stats.gamma.rvs(k, scale = 1/lamb) 
    else:
        return scp.stats.gamma.rvs(k, scale = 1/lamb, size=size) 

def log_to_normal(mu_log, sigma_log):
    """Finds the value of mu and sigma such that the lognormal distribution X ~ exp(mu + sigma*Z), with Z ~ Normal(0, 1), 
    has the desired mean mu_log and variance sigma_log"""
    
    mu = np.log(mu_log**2/np.sqrt(mu_log**2 + sigma_log**2))
    sigma = np.sqrt(np.log(1 + sigma_log**2/mu_log**2))
    
    return mu, sigma

def lognormal_matrix(N, mu=1., sigma=1.):
    """
    Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma.
    """
    
    # Choose parameters so that the lognormal has same variance and mean as the gaussian
    mu_normal, sigma_normal = log_to_normal(mu, sigma)
    mat = np.random.lognormal(mu_normal, sigma_normal, (N, N))

    return (mat + mat.conj().T)/np.sqrt(2)

def bernoulli_matrix(N, p=0.5):
    """
    Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma.
    """
    
    mat = np.random.binomial(n=1, p=p, size=(N, N))
    
    return mat + mat.conj().T

def gamma_dist_matrix(N, k, lamb=1):
    """Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma"""
    
    mat = gamma_dist(k, lamb, size=(N, N))
    
    return (mat + mat.conj().T)/np.sqrt(2)