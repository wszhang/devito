import numpy as np
import matplotlib 
from matplotlib import cm
from matplotlib import pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'

print("read iso")
f1 = open("data.iso.bin", "rb")
a1 = np.load(f1)
f1.close()

print("read iso_flatten")
f2 = open("data.iso_flatten.bin", "rb")
a2 = np.load(f2)
f2.close()

amax1 = np.max(np.abs(a1))
amax2 = np.max(np.abs(a2))
scale = 0.25
amin,amax = - scale * amax1, + scale * amax1

print("")
print("max iso;         %12.6f" % (amax1))
print("max iso_flatten; %12.6f" % (amax2))

nx,ny,nz = a1.shape
print(a1.shape)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.imshow(np.transpose(a1[:,:,nz//2+5]), cmap=cm.seismic, vmin=-amax, vmax=amax)
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("iso")

plt.subplot(1, 2, 2)
plt.imshow(np.transpose(a2[:,:,nz//2+5]), cmap=cm.seismic, vmin=-amax, vmax=amax)
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.title("iso_flatten")

plt.tight_layout()
plt.savefig("iso.png")
plt.show()
plt.close("all")