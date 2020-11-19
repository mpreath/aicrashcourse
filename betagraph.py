from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

a1, b1 = 10, 10
rv0 = beta(a1,b1)

a2, b2 = 3, 3
rv1 = beta(a2,b2)

x = np.linspace(0,1,100)
plt.plot(x,rv0.pdf(x))
plt.plot(x,rv1.pdf(x))
plt.show()