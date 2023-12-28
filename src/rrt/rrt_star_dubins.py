from matplotlib import pyplot as plt, axes
from scipy.spatial import KDTree
from shapely import Polygon
import numpy as np
import math
from typing import List
from src.utils import Point, deg_2_rad, Path
import timeit
from src.dubins.dubins_circle_to_point import calculate_dubins_path
from src.vehicle_dynamics import get_velocities, get_wheel_vel


def create_sample(i, dim):
    x = np.base_repr(i, base=2)[::-1]
    y = np.base_repr(i, base=3)[::-1]

    x_new, y_new = 0, 0

    for i, d in enumerate(x):
        x_new += int(d) / (2 ** (i+1))
    for i, d in enumerate(y):
        y_new += int(d) / (3 ** (i+1))

    return x_new * dim[0], y_new * dim[1]


def check_path_collision(obstacles, path: Path):
    is_collision = False

    for obstacle in obstacles:
        if path.intersects(obstacle):
            is_collision = True
            break

    return is_collision


def RRT(start: Point, goal: Point, obstacles: List[Polygon], dim, num_samples, vehicle_radius, dubins_radius, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            vehicle_radius, join_style="mitre"))

    for sample in range(num_samples):

        kdtree = KDTree([p.p for p in points])

        new_pt_coords = create_sample(sample, dim)

        proximity = (math.log(len(points)) / len(points)) * dim[0] * dim[1] / 2
        if proximity == 0:
            proximity = dim[0] * dim[1] / 2

        nearest_pt_idxs: List[int] = kdtree.query_ball_point(
            new_pt_coords, proximity)

        potential_parents = []
        potential_paths: List[Path] = []
        potential_distances = []

        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt = points[nearest_pt_idx]

            new_pt = Point(new_pt_coords, 0)
            dubins_path = calculate_dubins_path(
                nearest_pt, new_pt, dubins_radius)

            if not check_path_collision(buffered_obstacles, dubins_path):
                dist = dubins_path.length
                nearest_pt_distances.append(dist)

                new_dist = distances[nearest_pt_idx] + dist

                potential_parents.append(nearest_pt_idx)
                potential_paths.append(dubins_path)
                potential_distances.append(new_dist)

        if len(potential_paths) == 0:
            continue

        min_idx = np.argmin(potential_distances)
        dubins_path = potential_paths[min_idx]
        new_dist = potential_distances[min_idx]
        new_parent_idx = potential_parents[min_idx]

        new_pt = dubins_path.line.end_config

        points.append(new_pt)
        parents.append(new_parent_idx)
        distances.append(new_dist)

        new_pt_idx = len(points) - 1

        for remaining_pt_idx in nearest_pt_idxs:
            if remaining_pt_idx == new_parent_idx:
                continue

            remaining_pt = points[remaining_pt_idx]

            remaining_dubins_path: Path = calculate_dubins_path(
                new_pt, remaining_pt, dubins_radius)
            nearest_to_new_dist = remaining_dubins_path.length

            if new_dist + nearest_to_new_dist < distances[remaining_pt_idx]:
                if not check_path_collision(buffered_obstacles, remaining_dubins_path):
                    points[remaining_pt_idx].theta = remaining_dubins_path.line.end_config.theta
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

        dubins_path.plot(ax)

    kdtree = KDTree([p.p for p in points])

    total_dist = 0
    trajectory: List[Path] = []
    k = 1

    while k < len(points) - 1:
        dist, nearest_pt_idx = kdtree.query(goal.p, [k])
        dist, nearest_pt_idx = dist[0], nearest_pt_idx[0]
        nearest = points[nearest_pt_idx]
        path_to_goal = calculate_dubins_path(nearest, goal, dubins_radius)

        if check_path_collision(buffered_obstacles, path_to_goal):
            k += 1
        else:
            total_dist = distances[nearest_pt_idx] + dist
            trajectory.append(path_to_goal)

            while nearest != start:
                nearest_pt_idx = parents[nearest_pt_idx]
                parent: Point = points[nearest_pt_idx]

                path = calculate_dubins_path(parent, nearest, dubins_radius)
                trajectory.append(path)

                nearest = parent

            break

    return trajectory, total_dist


def setup_plot(dim, start: Point, goal: Point, obstacles, vehicle_radius):
    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, dim[0])
    ax.set_ylim(0, dim[1])
    ax.set_aspect('equal')

    init_circle = plt.Circle(start.p, vehicle_radius,
                             facecolor='none', edgecolor='b', zorder=15)
    ax.add_patch(init_circle)
    ax.plot(start.p[0], start.p[1], "bo", zorder=15)
    ax.plot(goal.p[0], goal.p[1], "go", zorder=15)

    for o in obstacles:
        x, y = o.exterior.xy
        ax.plot(x, y, "r")
        x, y = o.buffer(vehicle_radius, join_style="mitre").exterior.xy
        ax.plot(x, y, color="orange")

    return ax


if __name__ == "__main__":

    start = Point((1, 2), deg_2_rad(-90))
    goal = Point((8, 9), 0)

    obstacles = [
        Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        Polygon(((6, 6), (7, 5), (8, 6), (8, 8), (6, 8))),
        Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3)))
    ]

    NUM_SAMPLES = 300
    DIM = (10, 10)
    VEHICLE_RADIUS = 0.3
    DUBINS_RADIUS = 0.3

    ax = setup_plot(DIM, start, goal, obstacles, VEHICLE_RADIUS)

    start_time = timeit.default_timer()
    trajectory, dist = RRT(start, goal, obstacles, DIM,
                           NUM_SAMPLES, VEHICLE_RADIUS, DUBINS_RADIUS, ax)
    stop_time = timeit.default_timer()

    print("Elapsed time:", stop_time - start_time)

    trajectory_points = []

    for path in trajectory:
        path.plot(ax, color="g", zorder=10)
        ax.plot(path.curve1.end_config.p[0],
                path.curve1.end_config.p[1], "go", zorder=30)
        ax.plot(path.line.end_config.p[0],
                path.line.end_config.p[1], "go", zorder=30)

        trajectory_points[:0] = path.sample_points(ax)

    # print(len(trajectory_points))

    # for i, point in enumerate(trajectory_points):
    #     if i < len(trajectory_points) - 1:
    #         v, w = get_velocities(
    #             point, trajectory_points[i+1], 2 * VEHICLE_RADIUS)
    #         omega_l, omega_r = get_wheel_vel(v, w, 2 * VEHICLE_RADIUS)

    #         print("v:", v, " w:", w, " o_l:", omega_l, " o_r", omega_r)

    plt.show()
