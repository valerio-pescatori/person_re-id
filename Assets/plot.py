import matplotlib.pyplot as plt 
import sys
import numpy as np

# parsing y
y = sys.argv[1]
y = y.split(",")
y = [float(el) for el in y]
# parsing x
x = sys.argv[2]
x = [i for i in range(1, int(x) +1)]

# plotting
plt.plot(x,y)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5])
# plt.xticks(np.arange(0, 300, 20))
plt.xlabel("frame")
plt.ylabel("distance")
plt.show()