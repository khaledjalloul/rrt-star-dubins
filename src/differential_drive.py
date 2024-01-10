from matplotlib import pyplot as plt, axes
from shapely import Polygon, LinearRing, Point as ShapelyPoint
import numpy as np
import math
from src.utils import setup_rrt_plot, euc_distance, mod_pi, diff_dynamics
from src.classes import Point
from src.pid import PIDController
from src.rrt.rrt_star_dubins import RRT

def follow_path():
    start = Point(0, 9, 1)
    goal = ShapelyPoint(4.5, -2).buffer(0.5)

    NUM_SAMPLES = 200
    DIM = ((-11, -11), (11, 11))
    VEHICLE_RADIUS = 0.6
    DUBINS_RADIUS = 0.45
    NUM_PATH_SAMPLES = (10, 10)
    WHEELBASE = 1

    goal = goal.buffer(VEHICLE_RADIUS, join_style="mitre")
    g_minx, _, g_maxx, _ = goal.exterior.bounds
    g_center = goal.centroid
    g_radius = (g_maxx - g_minx) / 2 - 0.05

    goals = [Point(g_center.x, g_center.y, 0)]
    for i, radius in enumerate([g_radius / 3, 2 * g_radius / 3, g_radius]):
        th_range = np.linspace(0, 2 * np.pi, 15 - (i * 5))
        for th in th_range:
            x = radius * np.cos(th) + g_center.x
            y = radius * np.sin(th) + g_center.y

            goals.append(Point(x, y, th))

    obstacles = [
        Polygon(([-7, -7], [-7, -3], [-3, -3], [-3, -7], [-7, -7])),
        Polygon(([-1, -7], [-1, -3], [3, -3], [3, -7], [-1, -7])),
        Polygon(([5,  -7], [5, -3], [9, -3], [9, -7], [5, -7])),
        Polygon(([-7, -1], [-7, 3], [-3, 3], [-3, -1], [-7, -1])),
        Polygon(([-1, -1], [-1, 3], [3, 3], [3, -1], [-1, -1])),
        Polygon(([5, -1], [5, 3], [9, 3], [9, -1], [5, -1])),
        Polygon(([-6, 5], [-5, 6], [-4, 6], [-5, 4], [-6, 5])),
        Polygon(([0, 5], [1, 7], [2, 6], [1, 4], [0, 5])),
        Polygon(([6, 5], [7, 7], [8, 6], [7, 4], [6, 5])),
        LinearRing(([[-11, -11], [-11, 11], [11, 11], [11, -11], [-11, -11]]))
    ]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            VEHICLE_RADIUS, join_style="mitre"))

    pid = PIDController(0, 0, 2)

    current_x = start.x
    current_y = start.y
    current_th = start.theta

    w = 0

    ax: axes.Axes
    _, ax = plt.subplots()

    nearest_idx = 0

    dubins_paths, _ = RRT(
        start,
        goals,
        buffered_obstacles,
        DIM,
        NUM_SAMPLES,
        3 * NUM_SAMPLES,
        DUBINS_RADIUS,
        ax
    )

    rrt_path = []

    for path in dubins_paths:
        rrt_path[:0] = path.sample_points(NUM_PATH_SAMPLES)

    if len(rrt_path) == 0:
        return

    for i in range(10000):
        current = Point(current_x, current_y, current_th)
        ax.clear()
        setup_rrt_plot(DIM, start, goal, obstacles,
                       buffered_obstacles, VEHICLE_RADIUS, ax)
        ax.plot([p.x for p in rrt_path], [p.y for p in rrt_path], "y")
        ax.plot(current_x, current_y, "bo", zorder=100)

        next_point = rrt_path[nearest_idx]
        dist = euc_distance(current, next_point)

        while dist < 0.2 and nearest_idx < len(rrt_path) - 1:
            nearest_idx += 1
            next_point = rrt_path[nearest_idx]
            dist = euc_distance(current, next_point)

        if dist < 0.2 and nearest_idx == len(rrt_path) - 1:
            break

        theta_desired = math.atan2(
            (next_point.y - current.y), (next_point.x - current.x)
        )

        diff_theta = mod_pi(theta_desired - current.theta)
        acc_theta = pid.calculate(diff_theta, 0)

        v = (1 - math.e ** (- 0.7 * dist)) * math.e ** (- abs(diff_theta))
        w += acc_theta

        if w > 1:
            w = 1
        if w < -1:
            w = -1

        dx, dy, dth = diff_dynamics(v, current_th, w, WHEELBASE)

        current_x += dx
        current_y += dy
        current_th += dth

        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    follow_path()
