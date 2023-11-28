import numpy as np
from numpy import linalg as LA
import sympy as sym

# Initializing node info for Ku=f
N = 11
Ne=N-1
xl=0
xr=1
x=np.linspace(xl,xr,N)
u_x=np.sin(np.pi*x) # u(x,0)=sin(pi*x)
u_0=0 # u(0,t)=u(1,t)=0
t0=0
tf=1
dt=1/551

# Mass Matrix and reduced mass matrix
twos=N*[2]
ones=np.ones(Ne)
M=np.diag(twos)+np.diag(ones,k=-1)+np.diag(ones,k=1)
M[0][0]=0
M[Ne][Ne]=0
twos=(Ne-1)*[2]
ones=np.ones(Ne-2)
M_red=np.diag(twos)+np.diag(ones,k=-1)+np.diag(ones,k=1)
print(M)
print(M_red)