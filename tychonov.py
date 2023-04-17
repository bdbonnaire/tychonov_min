import numpy as np


def laplacian(v):
    """2D Laplacian with null Neumann condition on the boundaries.

    Args:
        v (np.array (N,N)): input image

    Returns: discrete laplacian of v, (np.array (N,N))

    """
    d1 = np.zeros_like(v)
    d2 = np.zeros_like(v)

    # Computations on the center
    d1[:, 1:-1] = 2*(v[:,2:] + v[:,:-2] - v[:,1:-1]) 
    d2[ 1:-1,:] = 2*(v[2:,:] + v[:-2,:] - v[1:-1,:])
    # Boundary conditions
    d1[:, 0] = 2*(v[:,1] - v[:,0])
    d2[ 0,:] = 2*(v[1,:] - v[0,:])
    d1[:, -1] = 2*(v[:,-2] - v[:,-1])
    d2[ -1,:] = 2*(v[-2,:] - v[-1,:])

    laplacian = d1 + d2
    return laplacian
