import numpy as np


# Initializing node info for Ku=f
N = 11
xl=0
xr=1
u_x=np.sin(np.pi*x) # u(x,0)=sin(pi*x)
u_0=0 # u(0,t)=u(1,t)=0

# Create uniform grid and connectivity map
Ne = N-1
h=(xr-xl)/Ne
x[N]=0
iee[Ne][2]=0
for i in Ne:
    x[i]=xl+(i-1)*h
    iee[i][0]=i
    iee[i][1]=i+1
x[N]=xr

# Define parent grid -1<C<1
phi1=(1-C)/2
phi2=(1+C)/2