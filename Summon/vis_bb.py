import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have the following variables defined:
# total_obj_list: List of bounding boxes
# scene_center: Numpy array representing the scene center

def fun(scene_center, total_obj_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the scene center
    ax.scatter(scene_center[0], scene_center[1], scene_center[2], c='red', label='Scene Center')

    # Plotting the bounding boxes and their half extents
    for bbox in total_obj_list:
        half_extent = bbox.get_half_extent()
        center = bbox.get_center()

        # Extracting the x, y, z coordinates from the center and half_extent arrays
        x, y, z = center[0], center[1], center[2]
        dx, dy, dz = half_extent[0], half_extent[1], half_extent[2]

        # Plotting the bounding box
        ax.add_artist(plt.Rectangle((x - dx, y - dy), 2 * dx, 2 * dy, facecolor='none', edgecolor='blue'))
        ax.plot([x - dx, x + dx], [y - dy, y - dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y - dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y - dy], [z + dz, z + dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z + dz, z + dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z + dz, z + dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z + dz, z + dz], 'blue')

        # Plotting the half extents as vectors
        ax.quiver(x, y, z, dx, dy, dz, color='green', label='Half Extent')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set legend
    ax.legend()

    # Show the plot
    plt.show()


