from matplotlib import pyplot as plt, axes
from scipy.spatial import KDTree
from shapely import Polygon, LinearRing
import numpy as np
import math
from typing import List
from src.utils import deg_2_rad, setup_rrt_plot, create_halton_sample
from src.classes import Point, Path
import timeit
from src.dubins.dubins_circle_to_point import calculate_dubins_path


def check_path_collision(obstacles, path: Path):
    is_collision = False

    for obstacle in obstacles:
        if path.intersects(obstacle):
            is_collision = True
            break

    return is_collision


def RRT(start: Point, goal: Point, obstacles: List[Polygon], dim, num_samples, dubins_radius, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]

    for sample in range(num_samples):

        kdtree = KDTree([p.tuple() for p in points])

        new_pt_coords = create_halton_sample(sample, dim)

        proximity = (math.log(len(points)) / len(points)) * \
            (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2
        if proximity == 0:
            proximity = (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2

        nearest_pt_idxs: List[int] = kdtree.query_ball_point(
            new_pt_coords, proximity)

        potential_parents = []
        potential_paths: List[Path] = []
        potential_distances = []

        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt = points[nearest_pt_idx]

            new_pt = Point(*new_pt_coords, 0)
            dubins_path = calculate_dubins_path(
                nearest_pt, new_pt, dubins_radius)

            if not check_path_collision(obstacles, dubins_path):
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
                if not check_path_collision(obstacles, remaining_dubins_path):
                    points[remaining_pt_idx].theta = remaining_dubins_path.line.end_config.theta
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

        dubins_path.plot(ax)

    kdtree = KDTree([p.tuple() for p in points])

    total_dist = 0
    trajectory: List[Path] = []
    k = 1

    while k < len(points) - 1:
        dist, nearest_pt_idx = kdtree.query(goal.tuple(), [k])
        dist, nearest_pt_idx = dist[0], nearest_pt_idx[0]
        nearest = points[nearest_pt_idx]
        path_to_goal = calculate_dubins_path(nearest, goal, dubins_radius)

        if check_path_collision(obstacles, path_to_goal):
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


if __name__ == "__main__":

    start = Point(1, 2, deg_2_rad(-90))
    goal = Point(8, 9, 0)

    NUM_SAMPLES = 400
    DIM = ((0, 0), (10, 10))
    VEHICLE_RADIUS = 0.3
    DUBINS_RADIUS = 0.3

    obstacles = [
        Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        Polygon(((6, 6), (7, 5), (8, 6), (8, 8), (6, 8))),
        Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3))),
        LinearRing(((0, 0), (10, 0), (10, 10), (0, 10)))
    ]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            VEHICLE_RADIUS, join_style="mitre"))

    ax = setup_rrt_plot(DIM, start, goal, obstacles,
                        buffered_obstacles, VEHICLE_RADIUS)

    start_time = timeit.default_timer()
    trajectory, dist = RRT(start, goal, buffered_obstacles,
                           DIM, NUM_SAMPLES, DUBINS_RADIUS, ax)
    stop_time = timeit.default_timer()

    print("Elapsed time:", stop_time - start_time)

    trajectory_points = []

    for path in trajectory:
        path.plot(ax, color="g", zorder=10)
        ax.plot(path.curve1.end_config.x,
                path.curve1.end_config.y, "go", zorder=30)
        ax.plot(path.line.end_config.x,
                path.line.end_config.y, "go", zorder=30)

        trajectory_points[:0] = path.sample_points(ax)

    # print(len(trajectory_points))

    # for i, point in enumerate(trajectory_points):
    #     if i < len(trajectory_points) - 1:
    #         v, w = get_velocities(
    #             point, trajectory_points[i+1], 2 * VEHICLE_RADIUS)
    #         omega_l, omega_r = get_wheel_vel(v, w, 2 * VEHICLE_RADIUS)

    #         print("v:", v, " w:", w, " o_l:", omega_l, " o_r", omega_r)

    plt.show()
