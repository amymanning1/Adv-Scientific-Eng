import numpy as np
from numpy import linalg as LA
import mysvd

# have user input number of springs and masses 
def take_input():
    num_mass = input('How many masses do you have?\n')
    num_spring = input('How many springs do you have?\n')
    if int(num_spring)-int(num_mass)>1 or int(num_spring)-int(num_mass)<0: raise Exception('Amount of masses and springs may only have a difference of one.')
    mass_list = []
    spring_list=[]
    i=1 
    j=1
    while i<int(num_mass)+1:
        mass = input('Enter mass' + str(i) + '\n')
        mass_list.append(float(mass))
        i+=1
    while j<int(num_spring)+1:
        spring = input('Enter spring constant' + str(j) + '\n')
        spring_list.append(float(spring))
        j+=1

    bound_quest=input('Please select boundary conditions (1): Fixed/Fixed, (2): Fixed/Free\n')
    if bound_quest == 1 or str(1):
        bound_cond = 'fixed/fixed'
        
    elif bound_quest == 2 or str(2):            
        bound_cond = 'fixed/free'
        
    else:    
        print('Please enter either 1 or 2')
        exit
    
    g=9.81 # gravity m/s^2



    mass_list = np.array(mass_list)
    f = g*mass_list # force vector
    C = np.diag(spring_list) # matrix of spring constants
    
    return num_mass, num_spring, f, C, bound_cond


def calc_elongation(A,u,num_spring):
    e = [0]*int(num_spring)
    e = A@u
    return np.array(e)



def main():
    num_mass, num_spring, f, C, bound_cond = take_input()
    
    # Find A
    A1 = (np.array([1]*int(num_mass))) 
    A2 = np.zeros(((int(num_mass)),int(num_spring)))
    A = np.zeros((int(num_spring),int(num_mass)))
    np.fill_diagonal(A,1)
    if num_mass == num_spring:
        neg_ones = np.diag([-1]*(int(num_mass)-1),k=-1)
    else:
        neg_ones = np.diag([-1]*int(num_mass),k=-1)
        neg_ones = np.delete(neg_ones, int(num_mass),axis=1)
    A=A+neg_ones

    # Find K
    I = np.identity((A.shape[0]))
    K = A.T@I@A
    print('Stiffness Matrix, K = ')
    print(K)

    # Solve force equation for displacements
    u = f*LA.inv(K) 
    print('Displacement matrix u = ')
    print(u)        
    e = calc_elongation(A,u,num_spring)
    print('Elongation = ')
    print(e)
    
    w = C@e # internal force equation
    print('w = ')
    print(w)

    eigenvalues_u, eigenvectors_u, eigenvalues_v, eigenvectors_v = mysvd.calc_eig(A)
    print('Eigenvalues: ' )
    print(eigenvalues_v)
    
    condNum = mysvd.calc_condNum(eigenvalues_v)
    print('Condition Number: ' + str(condNum) + '\n')
    U,V, sigma, A_inv = mysvd.svd_inv(eigenvalues_u, eigenvalues_v,eigenvectors_u,eigenvectors_v, A)
    print('Singular Values: ')
    print(sigma)



if __name__ == '__main__':
    main()
