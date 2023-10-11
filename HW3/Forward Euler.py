import numpy as np 
import matplotlib.pyplot as plot
t = [-1, 0, 0.00001, 0.01, 1, 1.0000006, 5.5, 100000000]
alpha = int(3)
#dydt = -1 * alpha * y
ylist=[]
for i in t:
    y = np.exp(-1*alpha * i) # Exact Solution
    yprime = (1-alpha*i)*y # Update Equation
    # print(y)
    # print(yprime)

# graph


