import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

ZONE = 5000
NZONE = -5000

def update_plot(num):
    ax.cla()
    ax.set_xlim3d(NZONE, ZONE)
    ax.set_ylim3d(NZONE, ZONE)
    ax.set_zlim3d(NZONE, ZONE)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Plot particle positions at current time step
    x = positions[num][:, 0]
    y = positions[num][:, 1]
    z = positions[num][:, 2]
    #ax.scatter(x, y, z, c='b')
    ax.scatter(x, y, z, c='b', s=10)
    
    # Set title to current time step
    ax.set_title(f"Time step {num}")

positions = []


# Get absolute path of directory containing the script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Number of iterations
numIterations = 1000

for i in range(numIterations):
    # Load position data from file
    filename = os.path.join(dir_path, f"positions_{i}.txt")
    data = np.loadtxt(filename)
    
    # Append data to list
    positions.append(data)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(-100, 100)
ax.set_ylim3d(-100, 100)
ax.set_zlim3d(-100, 100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot initial particle positions
x = positions[0][:, 0]
y = positions[0][:, 1]
z = positions[0][:, 2]
ax.scatter(x, y, z, c='b')



# Create animation object
ani = FuncAnimation(fig, update_plot, frames=numIterations, interval=1)

# Display animation
plt.show()

