import numpy as np

def is_unitary_matrix(U, tol=1e-6):
    """Verifies whether the matrix is unitary."""

    # Dimensions
    U = np.array(U)
    dims = U.shape

    # Sanity check for the dimensionality: needs to be a square matrix
    assert len(dims) == 2, "Array is not in matrix form"
    assert dims[0] == dims[1], "Array is not square matrix"

    # Identity matrix
    identity = np.eye(dims[0], dtype=U.dtype)
    
    return np.allclose(U.conj().T @ U, identity, atol=tol)
    
def is_dual_unitary_tensor(U, tol=1e-6):
    """Verifies dual-unitarity of a 4-index tensor U[i,j,k,l]. U must have dimensions (q, q, q, q)."""
    
    # Dimensions
    U = np.array(U)
    dims = U.shape
    q = dims[0]

    # Sanity check for the dimensionality: needs to be two-legged qudit gate
    assert len(dims) == 4, "Tensor is not two-legged"
    assert dims.count(q) == 4, "Legs do not have all the same dimensions"
    
    # Reshapes the dual and the original matrix
    U_dual = U.transpose(0, 2, 1, 3).reshape(q**2, q**2) # Reshape: (ik, jl)
    U      = U.reshape(q**2, q**2)                       # Reshape: (ij, kl)
    
    return is_unitary_matrix(U, tol=tol) and is_unitary_matrix(U_dual, tol=tol)