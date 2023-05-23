import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# Create axis
edge_length = 1
axes = [edge_length, edge_length, edge_length]

# Create Data
data = np.ones(axes, dtype=bool)

# Color
in_set_color = 'green'
out_set_color = 'lightgrey'

# Scale aproach
# colors = np.empty(axes, dtype=object)
# colors[:, :, :] = in_set_color

# Reality / Random Interpolation
# colors = np.empty(axes, dtype=object)
# surface_voxels = []
# for i in range(edge_length):
#     for j in range(edge_length):
#         surface_voxels.append([i, 0, j])
#         surface_voxels.append([edge_length - 1, i, j])
#         surface_voxels.append([i, j, edge_length - 1])
# random_in_set = random.sample(surface_voxels, 200)
# colors[:, :, :] = out_set_color
# for i in random_in_set:
#     colors[i[0], i[1], i[2]] = in_set_color

# Systematic Analysis 1 (interpolation within mechanism/hole-filling in)
# colors = np.empty(axes, dtype=object)
# colors[:, :, :] = in_set_color
# colors[:, 3:7, 5:9] = out_set_color
# colors[4:7, :,  5:9] = out_set_color
# colors[4:7, 3:7, :] = out_set_color

# Systematic Analysis 2 (non-random interpolation)
# colors = np.empty(axes, dtype=object)
# colors[:, :, :] = out_set_color
# colors[:, 1::2, :] = in_set_color
# colors[::2, :, :] = in_set_color

# Systematic Analysis 3 (extrapolation)
# colors = np.empty(axes, dtype=object)
# colors[:, :, :] = out_set_color
# colors[2:edge_length, 0, 2:edge_length] = in_set_color
# colors[-1, 0:edge_length-2, 2:edge_length] = in_set_color
# colors[2:edge_length, 0:edge_length-2, -1] = in_set_color

# Composition
colors = np.empty(axes, dtype=object)
colors[:, :, :] = out_set_color
colors[:, 0, 0] = in_set_color
colors[-1, :, 0] = in_set_color
colors[-1, 0, :] = in_set_color

# Set edge length = 1, to get 1x1x1 cube for legend
# colors = np.empty(axes, dtype=object)
# colors[:, :, :] = out_set_color
# # colors[:, :, :] = in_set_color

# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Hide grid lines
ax.grid(False)
# Hide axes
ax.set_axis_off()
# Add quiver
ax.quiver([10,10,10],[0,0,0],[-2.2,-2.2,-2.2],[-2,0,0],[0,2,0],[0,0,2], color='black', arrow_length_ratio=0.3)
# Axes labels for 10x10x10 cube
ax.text(6.7, 0, -2.6, 'M1', color='black')
ax.text(10, 2.1, -2.2, 'M2', color='black')
ax.text(8.7, 0, -1.3, 'M3', color='black')

ax.voxels(data, facecolors=colors, edgecolors='grey')

plt.tight_layout()
plt.show()