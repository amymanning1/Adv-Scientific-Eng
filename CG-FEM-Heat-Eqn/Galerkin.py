import numpy as np
from numpy import linalg as LA
import sympy as sym

# Initializing node info for Ku=f
N = 11
xl=0
xr=1
x=np.linspace(xl,xr,N)
u_x=np.sin(np.pi*x) # u(x,0)=sin(pi*x)
u_0=0 # u(0,t)=u(1,t)=0
#t=sym.Symbol('t')
# x=sym.Symbol('x')

def f(x,t):
    func = (np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x)
    return func

# BCs
nat_bound = [0, 1]
dbc = u_x

# initializing vars for Fe
# will use u_x from section above as initial condition
delta_t=1/551
T0=0
Tf=15 #seconds?
t=sym.Symbol('t')
#t=np.arange(T0,Tf,delta_t)

# Create uniform grid and connectivity map
Ne = N-1
h=(xr-xl)/Ne
x=np.zeros(N)
iee=np.zeros((Ne,2))
i=0
while i<Ne:
    x[i]=xl+(i-1)*h
    iee[i][0]=i
    iee[i][1]=i+1
    i+=1
x[Ne]=xr
dc=2/h #wrt x
dx= h/2 #wrt c


# Find M (mass matrix), nontrivial calculated in pdf both already reduced M* K*
twos=[2]*(Ne-1)
neg_ones=[-1]*(Ne-2)
M=np.diag(twos)+np.diag(neg_ones,k=-1)+np.diag(neg_ones,k=1)
M=M/30
invM=LA.inv(np.array(M))

# Build stiffness matrix
K=np.diag(twos)+np.diag(neg_ones,k=-1)+np.diag(neg_ones,k=1)
K=K/4


# Define and initialize parent grid -1<C<1
K=np.zeros((N,N)) # global stiffness
F=np.zeros(N) # global RHS
klocal=np.zeros((2,2)) # local element stiffness
flocal=np.zeros(2) # local element RHS

m=[0, 1]
l=[0, 1]
k=0
while k<Ne:
    # local element calculations done in pdf submitted
    flocal=[-0.2485208087*np.exp(-t), 0.2485208087*np.exp(-t)]
    klocal=[[2.5, -2.5],[2.5, 2.5]]
    
    # Finite Element assembly
    global_node1=iee[k][1]
    
    for j in m:
        F[global_node1] += flocal[l]
        for i in m:
            global_node2=iee[k][m]
            K[global_node1][global_node2]+=klocal[l][m]
    k+=1


# Natural BC pg 99
for i in l:
    if xl==nat_bound[i]:
        F[1]+= 0
    if xr ==nat_bound[i]:
        F[N]+=0
# DBC make list of dbc and check if element in list or not
for i in N:
    if i==dbc:
        for j in N:
            if i!=j:
                K[j][i]=0
                K[i][j]=0
                F[j]=F[j]-K[j][i]*dbc[i]
        F[i]=dbc[i]
        K[i][i]=1

# Solving Ku=f for u
func=f(x,t)
u=func*LA(np.array(K))

# Forward Euler Heat Equation Addition
n=0
k=0
l=0
nt=(Tf-T0)/delta_t
ctime=T0
quadpt=1/np.sqrt(3)
phi=[(1-quadpt)/2, (1+quadpt)/2]
p=f(quadpt,ctime)
g=f(-1*quadpt,ctime)
while n<nt:
    ctime=T0+(n+1)*delta_t
    n+=1
    # Build time dependent RHS vector
    while k<Ne:
        # 2nd order quadrature
        flocal[0]=p*(dc)*(phi[0])+g*(dc)*(phi[1])
        flocal[1]=p*(dc)*(phi[1])+g*(dc)*(phi[0])
        l=0
        while l<2:
            global_node1=iee[k][l]
            F[global_node1]+=flocal[l]
        k+=1

#u_n = u_n - delta_t*invM*K*u_n + delta_t*invM*F
    
# Backward Euler