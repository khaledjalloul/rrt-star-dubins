import numpy as np
import math
from typing import List
from matplotlib import pyplot as plt, axes
from utils import Point, deg_2_rad, euc_distance, Path

def get_wheel_vel(v, w, L):
    omega_l = v - w * (L / 2)
    omega_r = v + w * (L / 2)
    return omega_l, omega_r

def get_velocities(current: Point, next: Point):
    dy = next.p[1] - current.p[1]
    dx = next.p[0] - current.p[0]
    d_theta = next.theta - current.theta
    
    v = dx / math.cos(current.theta)

def dubins_dynamics(v, w, dt, start: Point, goal: Point) -> List[Point]:
    current = Point(start.p, start.theta)
    states = []

    distance_to_goal = euc_distance(current, goal)
    for i in range(100):
        x_dot = v * np.cos(current.theta)
        y_dot = v * np.sin(current.theta)
        theta_dot = w

        current.p = (current.p[0] + x_dot * dt, current.p[1] + y_dot * dt)
        current.theta += theta_dot * dt

        states.append(current)
        current = Point(current.p, current.theta)

    return states


if __name__ == "__main__":

    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    v = 1.0
    w = 0.5
    dt = 0.1
    
    start = Point((3, 3), deg_2_rad(90))
    goal = Point((7, 7), deg_2_rad(0))

    states: List[Point] = dubins_dynamics(v, w, dt, start, goal)

    ax.plot([s.p[0] for s in states], [s.p[1] for s in states])
    plt.show()
