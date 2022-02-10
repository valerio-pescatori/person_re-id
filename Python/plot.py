import matplotlib.pyplot as plt
import sys

# parsing y
y = sys.argv[1]
y = y.split(",")
y = [float(el) for el in y]
# parsing x
x = sys.argv[2]
x = [i for i in range(1, int(x) + 1)]

# plotting
plt.plot(x, y)
plt.xlabel("frame")
plt.ylabel("distance")
# plt.show()
plt.savefig("Python/Plots/plot " + sys.argv[3]+".png")
