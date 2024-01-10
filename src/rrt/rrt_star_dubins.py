from matplotlib import pyplot as plt, axes
from scipy.spatial import KDTree
from shapely import Polygon, LinearRing, Point as ShapelyPoint
import numpy as np
import math
from typing import List
from src.utils import setup_rrt_plot, create_halton_sample
from src.classes import Point, Path
from src.dubins.dubins_circle_to_point import calculate_dubins_path


def check_path_collision(obstacles, path: Path):
    is_collision = False

    for obstacle in obstacles:
        if path.intersects(obstacle):
            is_collision = True
            break

    return is_collision


def RRT(start: Point, goals: List[Point], obstacles: List[Polygon], dim, num_samples, max_samples, dubins_radius, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]
    total_dist = 0
    trajectory: List[Path] = []

    i = -1
    while len(trajectory) == 0 and num_samples < max_samples:
        while i < num_samples:
            i += 1
            kdtree = KDTree([p.tuple() for p in points])

            new_pt_coords = create_halton_sample(i, dim)

            proximity = (math.log(len(points)) / len(points)) * \
                (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2
            if proximity == 0:
                proximity = (dim[1][0] - dim[0][0]) * \
                    (dim[1][1] - dim[0][1]) / 2

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

            if ax is not None:
                dubins_path.plot(ax)

        for g in goals:
            point_tuples = np.array([[p.x, p.y] for p in points])
            goal_distances = (
                (point_tuples[:, 0] - g.x) ** 2 +
                (point_tuples[:, 1] - g.y) ** 2
            ) ** 0.5
            j = 0
            goal_distances = goal_distances + distances
            while j < len(points):
                nearest_pt_idx = np.argmin(goal_distances)
                total_dist = goal_distances[nearest_pt_idx]
                nearest = points[nearest_pt_idx]
                path_to_goal = calculate_dubins_path(nearest, g, dubins_radius)

                if check_path_collision(obstacles, path_to_goal):
                    goal_distances[nearest_pt_idx] = np.inf
                    j += 1
                else:
                    trajectory.append(path_to_goal)

                    while nearest != start:
                        nearest_pt_idx = parents[nearest_pt_idx]
                        parent: Point = points[nearest_pt_idx]

                        path = calculate_dubins_path(
                            parent, nearest, dubins_radius)
                        trajectory.append(path)

                        nearest = parent

                    break

            if len(trajectory) > 0:
                break

        num_samples += 100

    return trajectory, total_dist


if __name__ == "__main__":
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

    ax: axes.Axes
    _, ax = plt.subplots()

    setup_rrt_plot(DIM, start, goal, obstacles,
                    buffered_obstacles, VEHICLE_RADIUS, ax)

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

    ax.plot([p.x for p in rrt_path], [p.y for p in rrt_path], "g")
    plt.show()
