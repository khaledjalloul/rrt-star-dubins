from matplotlib import axes, pyplot as plt
from shapely import LinearRing
import math
import numpy as np
from src.classes import Point


def rad_2_deg(angle: float) -> float:
    return angle * 180 / math.pi


def mod_2_pi(x: float) -> float:
    return x - 2 * np.pi * np.floor(x / (2 * np.pi))


def mod_pi(x):
    if x == 0:
        return 0
    div, res = abs(x) // np.pi, abs(x) % np.pi
    res -= np.pi if div % 2 != 0 else 0
    res *= x / abs(x)
    return res


def deg_2_rad(angle: float) -> float:
    return angle * math.pi / 180


def euc_distance(a: Point, b: Point):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** (1 / 2)


def create_halton_sample(i, dim):
    x = np.base_repr(i, base=2)[::-1]
    y = np.base_repr(i, base=3)[::-1]

    x_new, y_new = 0, 0

    for i, d in enumerate(x):
        x_new += int(d) / (2 ** (i+1))
    for i, d in enumerate(y):
        y_new += int(d) / (3 ** (i+1))

    return x_new * (dim[1][0] - dim[0][0]) + dim[0][0], y_new * (dim[1][1] - dim[0][1]) + dim[0][1]


def setup_rrt_plot(dim, start: Point, goal: Point, obstacles, buffered_obstacles, vehicle_radius, ax: axes.Axes):

    ax.set_xlim(dim[0][0], dim[1][0])
    ax.set_ylim(dim[0][1], dim[1][1])
    ax.set_aspect('equal')

    start_circle = plt.Circle(start.tuple(), vehicle_radius,
                             facecolor='none', edgecolor='r', zorder=15)
    ax.add_patch(start_circle)
    ax.plot(start.x, start.y, "ro", zorder=15)
    ax.plot(goal.x, goal.y, "go", zorder=15)

    for i in range(len(obstacles)):
        if type(obstacles[i]) != LinearRing:
            x, y = obstacles[i].exterior.xy
            ax.plot(x, y, "r")
        x, y = buffered_obstacles[i].exterior.xy
        ax.plot(x, y, color="orange")
        
