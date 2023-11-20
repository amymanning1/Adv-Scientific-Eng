import numpy as np 
import matplotlib.pyplot as plt
x=np.arange(0,1,0.00001)
y=np.sin(np.pi*x)
plt.plot(x,y,color='blue',label='f(x)')

e1=np.arange(0,0.25, 0.0001)
phi1 = np.sin(np.pi/4)*e1
plt.plot(e1,phi1,color='red', label='fh')

e2=np.arange(0.25,0.5, 0.0001)
phi2 = np.sin(np.pi/2)*e2 + phi1
plt.plot(e2,phi2,color='red')

e3=np.arange(0.75,1, 0.0001)
phi3 = np.sin(3*np.pi/4)*e2 + phi1 + phi2
plt.plot(e3,phi3,color='red')

plt.legend()
plt.show()