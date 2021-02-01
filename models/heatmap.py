import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)
plt.rcParams["figure.figsize"] = 5,2

#x = np.linspace(-3,3)
y = np.cumsum(np.random.randn(50))+6

fig, ax = plt.subplots(nrows=1, sharex=True)

#extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto")
ax.set_yticks([])
#ax.set_xlim(extent[0], extent[1])

#plt.tight_layout()
plt.show()