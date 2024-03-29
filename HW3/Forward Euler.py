import numpy as np 
import matplotlib.pyplot as plt
delta_t = [0.0001, 0.01, 0.1, 1, 2, 5] # different step sizes
alpha = 0.5
for i in delta_t:
    t=np.arange(0,10,i)
    n=0
    ylist=[]
    exact_y =[]
    y0=1
    for j in t:    
        y = np.exp(-1 * alpha * j) # Exact Solution
        exact_y.append(y) # add another exact solution to list to plot
        if n==0:
            y_new = (1-alpha*i)*y0
            ylist.append(y_new)
        else : 
            y_old = ylist[n-1]
            y_new = (1-alpha*i)*y_old # Update Equation
            ylist.append(y_new) # Add new y guess to list
        n+=1
    # Graph
    plt.plot(t, exact_y, color='red', label='Exact Solution')
    plt.plot(t, ylist, color='blue', label='Forward Euler Estimate with delta t = ' + str(i))
    plt.title('Exact Solution vs Forward Euler')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()
      


