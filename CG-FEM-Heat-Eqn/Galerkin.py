import numpy as np
import sympy as sym
# Initializing node info for Ku=f
N = 11
xl=0
xr=1
x=np.linspace(xl,xr,N)
u_x=np.sin(np.pi*x) # u(x,0)=sin(pi*x)
u_0=0 # u(0,t)=u(1,t)=0
#fxt=(np.pi^2-1)*np.exp(-t)*np.sin(np.pi*x)
# BCs
nat_bound = [0, 1]
dbc = np.sin(np.pi* x)



# initializing vars for Fe
# will use u_x from section above as initial condition
delta_t=1/551
T0=0
Tf=15 #seconds?
t=sym.Symbol('t')
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

# Find M (mass matrix), nontrivial
c=sym.Symbol('c')
g=((1-c)/2)+((1+c)/2)+(h/2)
M=np.zeros((Ne,Ne))
i=0
j=0
while i<Ne:
    while j<Ne:
        M[i][j] = sym.integrate(g,(c, -1, 1)) 
        j+=1
    i+=1 


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

# Forward Euler Heat Equation Addition



    
