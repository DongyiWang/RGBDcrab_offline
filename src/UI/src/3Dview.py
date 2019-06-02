#!/usr/bin/env python 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('..//result//2//postures.pkl', 'rb') as f:
    postures = pickle.load(f)

# postures_xyz = []
# for i in range(postures.__len__()):
#    posture_single = postures[i]
#    posture_single = posture_single.position
#    postures_xyz.append([posture_single.x, posture_single.y, posture_single.z])

# postures_xyz = np.asarray(postures_xyz)
# print postures_xyz.shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(postures.__len__()):
    posture_single = postures[i]
    posture_single = posture_single.position
    ax.scatter(posture_single.x, posture_single.y, posture_single.z, marker='o', c='r')

ax.set_zlim(0, 1)
plt.show()
