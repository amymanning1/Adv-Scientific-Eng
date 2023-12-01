import numpy as np
from numpy import linalg as LA
import sympy as sym
import math as m
import matplotlib.pyplot as plt

def phi1(c):
    phi1=(1-c)/2
    return phi1

def phi2(c):
    phi2=(1+c)/2
    return phi2

def nontriv_mat(N,Ne,h):
    # Mass Matrix, reduced mass matrix, K, K_red, M_inv 
    twos=N*[2]
    fours=N*[4]
    ones=np.ones(Ne)
    M=np.diag(fours)+np.diag(ones,k=-1)+np.diag(ones,k=1)
    M[0][0]=0
    M[Ne][Ne]=0
    M=M*(h/6)
    K=np.diag(twos)+np.diag(-1*ones,k=-1)+np.diag(-1*ones,k=1)
    K[0][0]=1
    K[Ne][Ne]=1
    K=K/h
    # Now building reduced versions
    twos=(Ne-1)*[2]
    ones=np.ones(Ne-2)
    fours=(Ne-1)*[4]
    M_red=(np.diag(fours)+np.diag(ones,k=-1)+np.diag(ones,k=1))*(h/6)
    M_inv=LA.inv(M_red) # M_inv is inverse of reduced M matrix
    K_red=(np.diag(twos)+np.diag(-1*ones,k=-1)+np.diag(-1*ones,k=1))
    K_red=K_red/h
    return K_red,M_inv,M_red

def gridMap(Ne,xl,xr,h,x):
    i=0
    iee=np.zeros((Ne,2))
    while i<Ne:
        x[i]=xl+(i)*h
        iee[i][0]=i
        iee[i][1]=i+1
        i+=1
    x[Ne]=xr
    return x,iee

def dbc(z):
    ux=np.sin(np.pi*z)
    return ux

def forEul(nt,t0,dt,Ne,h,iee,x,M_inv,K_red,x_red,N):
    # Used quadrature to integrate F, work is in pdf
    # Now performing local element calculations
    n=1
    dx=h/2 #dx/dc
    u=dbc(x_red)
    while n<=nt:
        F=np.zeros(N) 
        ctime=t0+n*dt
        k=0
        # Build time dependent RHS vector
        while k<Ne-1: # loop over grid elements
            # 2 local nodes for every 1D element, 2nd order gaussian quadrature
            q1=1/m.sqrt(3) # Quadrature points, weights=1 in second order quadrature; if user chooses a higher order, initialize weights and intro in equation; here they are redundant
            q2=-1/m.sqrt(3)
            q1map=(q1+1)*dx+x[k]
            q2map=(q2+1)*dx+x[k]
            flocal=[0,0]
            flocal[0]=(np.pi**2-1)*np.exp(-ctime)*dx*(phi1(q1)*np.sin(np.pi*q1map)+phi1(q2)*np.sin(np.pi*q2map))
            flocal[1]=(np.pi**2-1)*np.exp(-ctime)*dx*(phi2(q1)*np.sin(np.pi*q1map)+phi2(q2)*np.sin(np.pi*q2map))
            
            # Finite Element Assembly
            l=0
            while l<2:
                global_node1=iee[k][l]
                F[int(global_node1)]+=flocal[l]
                l+=1
            k+=1
        # forward euler
        F_red=F[1:-1]
        u=u-dt*M_inv@K_red@u+dt*M_inv@F_red
        n+=1
    return u

def backEul(nt,t0,dt,Ne,h,iee,x,K_red,x_red,M_red,N,M_inv):
    # Used quadrature to integrate F, work is in pdf
    # Now performing local element calculations
    n=1
    dx=h/2
    u=dbc(x_red)
    while n<=nt:
        F=np.zeros(N)
        ctime=t0+n*dt
        k=0
        # Build time dependent RHS vector
        while k<Ne-1: # loop over grid elements
            # 2 local nodes for every 1D element, 2nd order gaussian quadrature
            q1=1/m.sqrt(3) # Quadrature points, weights=1 in second order quadrature; if user chooses a higher order, initialize weights and intro in equation; here they are redundant
            q2=-1/m.sqrt(3)
            q1map=(q1+1)*dx+x[k]
            q2map=(q2+1)*dx+x[k]
            flocal=[0,0]
            flocal[0]=(np.pi**2-1)*np.exp(-ctime+dt)*(h/2)*(phi1(q1)*np.sin(np.pi*q1map)+phi1(q2)*np.sin(np.pi*q2map))
            flocal[1]=(np.pi**2-1)*np.exp(-ctime+dt)*(h/2)*(phi2(q1)*np.sin(np.pi*q1map)+phi2(q2)*np.sin(np.pi*q2map))
            
            # Finite Element Assembly
            l=0
            while l<2:
                global_node1=iee[k][l]
                F[int(global_node1)]+=flocal[l]
                l+=1
            k+=1
        # backward euler
        F_red=F[1:-1]
        B=dt*M_red+K_red
        invB=LA.inv(B)
        # u=dt*invB@M_red@u + invB@F_red
        u=u-dt*M_inv@K_red@u+dt*M_inv@F_red
        n+=1
    return u

def main():
    # Initializing constants
    N = 11
    Ne=N-1
    xl=0
    xr=1
    h=(xr-xl)/Ne
    x=np.linspace(xl,xr,N)
    x_red=x[1:-1]
    t0=0 #time constants
    tf=1
    dt=1/600
    nt=(tf-t0)/dt
    
    # Call Functions
    K_red,M_inv,M_red=nontriv_mat(N,Ne,h)
    x,iee=gridMap(Ne,xl,xr,h,x)
    u=forEul(nt,t0,dt,Ne,h,iee,x,M_inv,K_red,x_red,N)
    v=backEul(nt,t0,dt,Ne,h,iee,x,K_red,x_red,M_red,N,M_inv)
    # Add natural bcs to u
    nat0=0 # u(0,t)=u(1,t)=0
    nat1=0
    u=np.append(u,nat1)
    u=np.insert(u,0,nat0)
    v=np.append(v,nat1)
    v=np.insert(v,0,nat0)
    
    # Plotting
    t=1
    exact_sol=np.exp(-t)*np.sin(np.pi*x)
    plt.plot(x,exact_sol,color='red',label='Exact Solution')
    plt.plot(x,u,color='blue',label='Forward Euler')
    plt.plot(x,v,color='green',label='Backward Euler')
    plt.title('Comparison with timestep ' + str(dt))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


            