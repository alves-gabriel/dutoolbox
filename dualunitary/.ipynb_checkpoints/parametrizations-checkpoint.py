from dataclasses import dataclass
import numpy as np
import scipy as scp
from scipy.linalg import expm
import utils.pauli as pauli
import randmatrix as rmt

# Data class for local onsite gates in DU
@dataclass
class onsite_gates:
    u_plus: np.ndarray
    u_min:  np.ndarray
    v_plus: np.ndarray
    v_min:  np.ndarray
        
def dual_unitary_gate_qubit(J_z, tau=np.pi/4, local_perturbation=None, seed=None
                            , u_plus=None, u_min=None, v_plus=None, v_min=None
                            , H_plus=None, H_min=None):
    r""" 
    Generates a (qubit) dual-unitary gate, following the notation:


        |           |
      ____         ____
     | u+ |       | u- |     ---->  On-site unitary
     |____|       |____|
         \ _______ /
          |       |
          |   V   |           ---->  XXZ gate
          |_______|
         /         \
      ____          ____
     | v+ |        | v- |     ---->  On-site unitary
     |____|        |____|
        |            |


    or, in the presence of local perturbations:


        |           |
      ____         ____
     | H+ |       | H- | 
     |____|       |____|     -----> On-site perturbation breaking integrability
         \ _______ /
          |       |
          |   V   |           
          |_______|
         /         \


    Reference: https://github.com/PieterWClaeys/UnitaryCircuits/blob/master/MaximumVelocityQuantumCircuits.ipynb

    Parameters
    ----------
    
    J_z : float 
        Anisotropy parameter
    
    tau : float 
        Trotter step. Corresponds to a dual-unitary gate at np.pi/4.
        
    local_perturbation : float
        Integrability-breaking perturbation (leads to interacting integrability if zero).

    seed: int
        Seed for random realization.
        
    u_plus, u_min, v_plus, v_min: array
        On-site unitaries. 
        If set to none, gates are randomly chosen based on site.
        
    H_plus, H_min:
        On-site Hamiltonians if a local perturbation is present. 
        If set to none, gates are randomly chosen based on site.

    Returns
    -------
    
        A dual-unitary gate in tensor form.
    """

    # Random realization seed
    if seed is not None:
        np.random.seed(seed)

    # Parametrization of entangling two-qubit gate
    M = tau*(_kronecker_tensor_form(pauli.X, pauli.X) + _kronecker_tensor_form(pauli.Y, pauli.Y) + J_z * _kronecker_tensor_form(pauli.Z, pauli.Z))
    V = expm(-1j * M).reshape([2, 2, 2, 2])
    
    # On-site unitaries
    local = onsite_gates(v_plus, v_min, u_plus, u_min)

    # Implements either parametrization
    if local_perturbation is None:
        
        # Picks a random unitary gates if not set
        for gate_name in ['u_plus', 'u_min', 'v_plus', 'v_min']:
            gate = getattr(local, gate_name)
            setattr(local, gate_name, _set_matrix("CUE", gate))
    else:
        
        # Picks a random Hamiltonian if not set
        H_plus, H_min = _set_matrix("GUE", H_plus), _set_matrix("GUE", H_min)

        # Sets onsite unitaries
        local.u_plus, local.u_min = expm(-1j*local_perturbation*H_plus), expm(-1j*local_perturbation*H_min)
        local.v_min, local.v_plus = np.eye(2), np.eye(2)

    # Full parametrization in tensor notation
    return np.einsum('ac, bd, cdef, eg, fh -> abgh', local.u_plus, local.u_min, V, local.v_plus, local.v_min)

###################
# HELPER FUNCIONS #
###################

def _kronecker_tensor_form(A, B):
    """Reshape kronecker products to tensor form."""
    return np.einsum('ij, kl -> ikjl', A, B).reshape(np.shape(A)[0]*np.shape(B)[0], np.shape(A)[1]*np.shape(B)[1])

def _set_matrix(m_type, matrix, dim=2, seed=None):
    """Helper function for picking a random matrix in case it is not chosen. This is flagged when the variable is 'None'"""
    
    # If the matrix is undetermined, sets a random realization
    if matrix is None:
        if seed is not None:
            np.random.seed(seed)

        # Ensemble type
        if m_type == "CUE":
            matrix = rmt.CUE_matrix(dim)
        elif m_type == "GUE":
            matrix = rmt.GUE_matrix(dim)

    # If the matrix is determined, the function did nothing
    return matrix