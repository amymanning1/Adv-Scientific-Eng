import numpy as np
from numpy import linalg as LA
import sympy as sym
import math as m

# Initializing constants
N = 11
Ne=N-1
xl=0
xr=1
h=(xr-xl)/Ne
x=np.linspace(xl,xr,N)
u_x=np.sin(np.pi*x) # u(x,0)=sin(pi*x)
u_0=0 # u(0,t)=u(1,t)=0
t0=0
tf=1
dt=1/551
c=sym.Symbol('c')
phi1=(1-c)/2
phi2=(1+c)/2
dc=2/h
dx=h/2
nt=(tf-t0)/dt
q1=1/m.sqrt(3) # Quadrature points, weights=1 in second order quadrature; if user chooses a higher order, initialize weights and intro in equation; here they are redundant
q2=-1/m.sqrt(3)

# Mass Matrix, reduced mass matrix, K, K_red, M_inv 
twos=N*[2]
ones=np.ones(Ne)
M=np.diag(twos)+np.diag(ones,k=-1)+np.diag(ones,k=1)
M[0][0]=0
M[Ne][Ne]=0
M=M*(h/6)
M_inv=LA.inv(M)
K=np.diag(twos)+np.diag(-1*ones,k=-1)+np.diag(-1*ones,k=1)
K[0][0]=1
K[Ne][Ne]=1
K=K/h
# Now building reduced versions
twos=(Ne-1)*[2]
ones=np.ones(Ne-2)
M_red=(np.diag(twos)+np.diag(ones,k=-1)+np.diag(ones,k=1))*(h/6)
K_red=(np.diag(twos)+np.diag(-1*ones,k=-1)+np.diag(-1*ones,k=1))/6

# Used quadrature to integrate F, work is in pdf
# Now performing local element calculations
n=1
k=0
l=0
while n<=nt:
    ctime=t0+n*dt
    n+=1
    # Build time dependent RHS vector
    for k<Ne: # loop over grid elements
        for l<2: # 2 local nodes for every 1D element
            