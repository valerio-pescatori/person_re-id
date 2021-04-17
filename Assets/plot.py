import matplotlib.pyplot as plt 
import sys

# parsing y
y = sys.argv[1]
y = y.split(",")
y = [float(el) for el in y]
# parsing x
x = sys.argv[2]
x = [i for i in range(1, int(x) +1)]

# plotting
plt.plot(x,y)
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5])
plt.xticks([0, 30, 50, 70, 100, 130, 150, 170, 200, 230, 250, 270, 300])
plt.xlabel("frame")
plt.ylabel("distance")
plt.show()