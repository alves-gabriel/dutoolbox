import numpy as np
import scipy as scp

def GUE_matrix(N):
    """Generates a random N x N matrix from the Gaussian unitary ensemble with unit variance at the diagonals."""

    # Random complex normal
    GUE = np.random.normal(0, 1, size=(N, N)) + 1j*np.random.normal(0, 1, size=(N, N))
    
    # Symmetrizes
    return (GUE + GUE.conj().T)/np.sqrt(2)

def CUE_matrix(N):
    return scp.stats.unitary_group.rvs(N)