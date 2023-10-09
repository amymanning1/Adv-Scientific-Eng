import array, cmath
import numpy as np 
from numpy import linalg as LA


# calculate eigenvalues/vectors
def calc_eig(A):
    eigenvalues_u, eigenvectors_u = LA.eig(A@A.T)
    eigenvalues_v, eigenvectors_v = LA.eig(A.T@A)
    # Sort eigenvalues and vectors greatest to least
    idx = eigenvalues_u.argsort()[::-1]   
    eigenvalues_u = eigenvalues_u[idx]
    eigenvectors_u = eigenvectors_u[:,idx]
    idy = eigenvalues_v.argsort()[::-1]   
    eigenvalues_v = eigenvalues_v[idy]
    eigenvectors_v = eigenvectors_v[:,idy]
    return eigenvalues_u, eigenvectors_u, eigenvalues_v, eigenvectors_v



# SVD inverse Ainv=VsigmainverseUtranspose
def svd_inv(eigenvalues_u, eigenvalues_v,eigenvectors_u,eigenvectors_v, A):
    # if any of the singular values are zero, return an error, matrix is not invertible
    if any((eigenvalues_v==0)):        
        raise Exception('Matrix is not invertible. Cannot perform SVD')
    else:
        U = eigenvectors_u
        V = eigenvectors_v
        same_sign = np.sign((A @ V)[0][0] * (U @ np.diag(eigenvalues_u))[0][0])
        V = V * same_sign.reshape(1, -1)
        sigma = np.zeros((A.shape[0], A.shape[1]))
        
        np.fill_diagonal(sigma,np.sqrt(eigenvalues_v.real))
        
    i=1
    for i in sigma:
        if (np.diag(sigma)).any()==0: 
            raise Exception('ERROR: A cannot be inverted. One or more singular values is equal to 0\n')
        else:    
            sigma_inv = np.zeros((sigma.shape[1], sigma.shape[0]))
            sigma_inv_diag = 1 / np.diag(sigma)
            np.fill_diagonal(sigma_inv, sigma_inv_diag)
            A_inv = (V@sigma_inv)@U.T
    return U, V, sigma, A_inv

# condition number
def calc_condNum(eigenvalues_v):
    max_eig = eigenvalues_v[0]
    sigma_max = np.sqrt(max_eig)
    min_eig = eigenvalues_v[-1]
    sigma_min = cmath.sqrt(min_eig)
    condNum = (sigma_max)/(sigma_min)
    return condNum

def main():
    num_list=[[1, 2],[3, 4]]
    A = np.array(num_list)
    eigenvalues_u, eigenvectors_u, eigenvalues_v, eigenvectors_v = calc_eig(A)
    U,V,sigma,A_inv = svd_inv(eigenvalues_u, eigenvalues_v, eigenvectors_u, eigenvectors_v,A)

    print('SVD Matrices:\n')
    print('A: ' + str(A) + '\n')
    print('U: ' + str(U) + '\n')
    print('Sigma: ' + str(sigma) + '\n')
    print('V^T: ' + str(V.T) + '\n')
    
    condNum = calc_condNum(eigenvalues_v)
    print('Condition Number: ')
    print(condNum)
    print('A^-1 = ')
    print(A_inv)
    

if __name__ == '__main__':
    main()
