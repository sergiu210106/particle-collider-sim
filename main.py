from typing import Union
from fastapi import FastAPI
import random
import io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
from fastapi.responses import StreamingResponse
import numpy as np

app = FastAPI()
batch_points = []

#Simulation parameters
BOUNDARY = 10
COLLISION_DAMPING = 0.9
FPS = 30


def generate_random_points(num_points=1000, batches=30):
    """Generate 3D points incrementally within a unit sphere."""
    batch_size = num_points // batches
    for _ in range(batches):
        for _ in range(batch_size):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            z = random.uniform(-1, 1)

            batch_points.append((x, y, z))
        yield batch_points


# Color mapping based on speed
def calculate_color(speeds):
    # Normalize speeds to the range [0, 1]
    normalized_speeds = np.clip(speeds / np.max(speeds), 0, 1)

    # Define RGB values for violet and red
    violet = np.array([148, 0, 211]) / 255  # RGB for violet
    red = np.array([255, 0, 0]) / 255  # RGB for red

    # Interpolate between violet and red
    colors = (normalized_speeds[:, None]) * violet + (1 - normalized_speeds[:, None]) * red
    return colors

# Generate initial positions and velocities
def generate_points(num_points):
    np.random.seed(42)
    positions = np.random.uniform(-BOUNDARY, BOUNDARY, (num_points, 2))
    velocities = np.random.normal(0, 0.5, (num_points, 2)) * np.linalg.norm(positions, axis=1).reshape(-1, 1)
    return positions, velocities


def update_positions(positions, velocities):
    positions += velocities / FPS

    # Wall collision detection and velocity adjustment
    for i in range(len(positions)):
        if abs(positions[i, 0]) >= BOUNDARY:
            velocities[i, 0] *= -COLLISION_DAMPING
        if abs(positions[i, 1]) >= BOUNDARY:
            velocities[i, 1] *= -COLLISION_DAMPING

    # Point collision detection and response (tangential bounce)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Calculate the distance vector between points i and j
            delta_pos = positions[j] - positions[i]
            dist = np.linalg.norm(delta_pos)

            # Check if points are close enough to collide
            if dist < 0.5 and dist != 0:
                # Normalize the distance vector to get the collision normal
                collision_normal = delta_pos / dist

                # Project velocities onto the collision normal
                v_i_normal = np.dot(velocities[i], collision_normal)
                v_j_normal = np.dot(velocities[j], collision_normal)

                # Exchange the normal components of the velocities (elastic collision)
                velocities[i] += (v_j_normal - v_i_normal) * collision_normal
                velocities[j] += (v_i_normal - v_j_normal) * collision_normal


# Function to create and save the animated GIF
def plot_simulation(num_points):
    # Generate points and initialize the plot
    positions, velocities = generate_points(num_points)
    fig, ax = plt.subplots()
    ax.set_xlim(-BOUNDARY, BOUNDARY)
    ax.set_ylim(-BOUNDARY, BOUNDARY)
    scat = ax.scatter([], [], s=20)

    # Define the init function
    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    # Update function for animation
    def update(frame):
        nonlocal positions, velocities
        update_positions(positions, velocities)

        # Calculate color based on speed
        speeds = np.linalg.norm(velocities, axis=1)
        colors = calculate_color(speeds)

        scat.set_offsets(positions)
        scat.set_color(colors)
        return scat,

    # Create and save the animation as GIF
    ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, interval=1000 / FPS, blit=True)

    filename = "simulate.gif"
    ani.save(filename, writer="pillow", fps=FPS)
    plt.close(fig)
    return filename

# Create animated GIF of the 3D points
def plot_points_animated(point_generator):
    global batch_points
    batch_points = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    inside_scatter = ax.scatter([], [], [], color='pink')
    outside_scatter = ax.scatter([], [], [], color='purple')

    def update(batch_points):
        inside_x, inside_y, inside_z = [], [], []
        outside_x, outside_y, outside_z = [], [], []
        for x, y, z in batch_points:
            if x ** 2 + y ** 2 + z ** 2 <= 1:
                inside_x.append(x)
                inside_y.append(y)
                inside_z.append(z)
            else:
                outside_x.append(x)
                outside_y.append(y)
                outside_z.append(z)

        inside_scatter._offsets3d = (inside_x, inside_y, inside_z)
        outside_scatter._offsets3d = (outside_x, outside_y, outside_z)
        return inside_scatter, outside_scatter

    ani = animation.FuncAnimation(fig, update, frames=point_generator, interval=50, repeat=True)
    filename = "anim.gif"
    # Save the animation to a GIF file
    ani.save(filename, writer="pillow", fps=15)
    plt.close(fig)
    return filename


# Endpoint to return the animated plot
@app.get("/plot/{num_points}")
async def get_plot(num_points: int = 1000):
    print(num_points, "aaaaaaaaaaaaaa")
    global batch_points
    batch_points = []
    point_generator = generate_random_points(num_points)
    gif_file = plot_points_animated(point_generator)
    # Open the GIF file in binary mode and stream it
    return StreamingResponse(open(gif_file, "rb"), media_type="image/gif")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/1012/1")
def read_root():
    return {"secret": "1287492ujfskdn@#$T$WGRFSV"}


@app.get("/items/{a}/{b}")
def read_item(a: int, b: int, q: Union[str, None] = None):
    return {"solutie": a + b, "q": q}

# FastAPI endpoint to return the animated plot
@app.get("/simulate")
async def get_plot(num_points: int = 100):
    gif_file = plot_simulation(num_points)
    return StreamingResponse(open(gif_file, "rb"), media_type="image/gif")