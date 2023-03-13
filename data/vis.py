import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

ZONE = 10000
NZONE = 3000

# Variables for camera position and zoom level
cam_x = 0
cam_y = 0
cam_z = 0
zoom_level = 1

def update_plot(num):
    global cam_x, cam_y, cam_z, zoom_level

    ax.cla()

    # Adjust axis limits based on camera position and zoom level
    ax.set_xlim3d(cam_x - ZONE / zoom_level, cam_x + ZONE / zoom_level)
    ax.set_ylim3d(cam_y - ZONE / zoom_level, cam_y + ZONE / zoom_level)
    ax.set_zlim3d(cam_z - ZONE / zoom_level, cam_z + ZONE / zoom_level)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot particle positions at current time step
    x = positions[num][:, 0]
    y = positions[num][:, 1]
    z = positions[num][:, 2]
    ax.scatter(x, y, z, c='b', s=10)

    # Set title to current time step
    ax.set_title(f"Time step {num}")

# Get absolute path of directory containing the script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Number of iterations
numIterations = 2000

# Load position data from files
positions = []
for i in range(numIterations):
    filename = os.path.join(dir_path, f"positions_{i}.txt")
    data = np.loadtxt(filename)
    positions.append(data)

# Create figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot initial particle positions
x = positions[0][:, 0]
y = positions[0][:, 1]
z = positions[0][:, 2]
ax.scatter(x, y, z, c='b', s=10)

# Set initial axis limits
ax.set_xlim3d(cam_x - ZONE, cam_x + ZONE)
ax.set_ylim3d(cam_y - ZONE, cam_y + ZONE)
ax.set_zlim3d(cam_z - ZONE, cam_z + ZONE)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Define keyboard controls for camera movement and zoom
def on_key_press(event):
    global cam_x, cam_y, cam_z, zoom_level
    if event.key == 'left':
        cam_x -= ZONE / 10
    elif event.key == 'right':
        cam_x += ZONE / 10
    elif event.key == 'down':
        cam_y -= ZONE / 10
    elif event.key == 'up':
        cam_y += ZONE / 10
    elif event.key == '+':
        zoom_level += 2
    elif event.key == '-':
        zoom_level -= 2
    elif event.key == 'a':
        cam_z -= ZONE / 10   # Move camera along z-axis
    elif event.key == 'z':
        cam_z += ZONE / 10   # Move camera along z-axis
    ax.set_xlim3d(cam_x - ZONE / zoom_level, cam_x + ZONE / zoom_level)
    ax.set_ylim3d(cam_y - ZONE / zoom_level, cam_y + ZONE / zoom_level)
    ax.set_zlim3d(cam_z - ZONE / zoom_level, cam_z + ZONE / zoom_level)
    fig.canvas.draw()

# Connect keyboard controls to figure
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Create animation object
ani = FuncAnimation(fig, update_plot, frames=numIterations, interval=1)

# Display animation
plt.show()

