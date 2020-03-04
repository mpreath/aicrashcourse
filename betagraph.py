from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

a1 = 2
b1 = 2

a2 = 7781
b2 = 1430

x = np.linspace(0,1.0,100)
y1 = beta.pdf(x,a1,b1)
y2 = beta.pdf(x,a2,b2)
plt.plot(x,y1,"-",x,y2,"r--")
plt.show()