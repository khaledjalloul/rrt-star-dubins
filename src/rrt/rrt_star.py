from matplotlib import axes, pyplot as plt
from scipy.spatial import KDTree
from shapely import LineString, Polygon, LinearRing, Point as ShapelyPoint
import numpy as np
import math
import timeit
from src.utils import setup_rrt_plot, euc_distance, create_halton_sample
from src.classes import Point
from typing import List


def check_line_collision(obstacles: List[Polygon], point1: Point, point2: Point):
    is_collision = False

    for obstacle in obstacles:
        if obstacle.intersects(LineString((point1.tuple(), point2.tuple()))):
            is_collision = True
            break

    return is_collision


def RRT(start: Point, goals: List[Point], obstacles, dim, num_samples, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]
    total_dist = 0
    trajectory: List[Point] = []

    for sample in range(num_samples):
        kdtree = KDTree([p.tuple() for p in points])

        new_pt = Point(*create_halton_sample(sample, dim), 0)

        proximity = (math.log(len(points)) / len(points)) * \
            (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2
        if proximity == 0:
            proximity = (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2

        nearest_pt_idxs: list = kdtree.query_ball_point(
            new_pt.tuple(), proximity)

        potential_parents = []
        potential_distances = []
        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt = points[nearest_pt_idx]

            dist = euc_distance(nearest_pt, new_pt)
            nearest_pt_distances.append(dist)

            new_dist = distances[nearest_pt_idx] + dist

            if not check_line_collision(obstacles, nearest_pt, new_pt):
                potential_parents.append(nearest_pt_idx)
                potential_distances.append(new_dist)

        if len(potential_distances) == 0:
            continue

        min_idx = np.argmin(potential_distances)
        new_dist = potential_distances[min_idx]
        new_parent_idx = potential_parents[min_idx]

        points.append(new_pt)
        parents.append(new_parent_idx)
        distances.append(new_dist)

        new_pt_idx = len(points) - 1

        for i, remaining_pt_idx in enumerate(nearest_pt_idxs):
            if remaining_pt_idx == new_parent_idx:
                continue

            remaining_pt = points[remaining_pt_idx]
            nearest_to_new_dist = nearest_pt_distances[i]

            if new_dist + nearest_to_new_dist < distances[remaining_pt_idx]:
                if not check_line_collision(obstacles, remaining_pt, new_pt):
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

        ax.plot([new_pt.x, points[potential_parents[min_idx]].x], [
            new_pt.y, points[potential_parents[min_idx]].y], "y-")

    for g in goals:
        point_tuples = np.array([[p.x, p.y] for p in points])
        goal_distances = (
            (point_tuples[:, 0] - g.x) ** 2 +
            (point_tuples[:, 1] - g.y) ** 2
        ) ** 0.5
        goal_distances = goal_distances + distances
        j = 0
        while j < len(points):
            nearest_pt_idx = np.argmin(goal_distances)
            total_dist = goal_distances[nearest_pt_idx]
            nearest = points[nearest_pt_idx]
            if check_line_collision(obstacles, g, nearest):
                goal_distances[nearest_pt_idx] = np.inf
                j += 1
            else:
                total_dist = distances[nearest_pt_idx] + dist
                trajectory.extend([g, nearest])

                while nearest != start:
                    nearest_pt_idx = parents[nearest_pt_idx]
                    parent: Point = points[nearest_pt_idx]

                    trajectory.append(parent)

                    nearest = parent

                break

        if len(trajectory) > 0:
            break

    return trajectory, total_dist


if __name__ == "__main__":

    start = Point(0, 9, 0)
    goal = ShapelyPoint(-6.9, 2.9).buffer(0.5)

    NUM_SAMPLES = 400
    DIM = ((-11, -11), (11, 11))
    VEHICLE_RADIUS = 0.6

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
        Polygon(([-7, -1], [-7, 3], [-1, 3], [-1, -1], [-7, -1])),
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
            VEHICLE_RADIUS + 0.05, join_style="mitre"))

    fig, ax = plt.subplots()

    setup_rrt_plot(DIM, start, goal, obstacles,
                   buffered_obstacles, VEHICLE_RADIUS, ax)

    start_time = timeit.default_timer()
    trajectory, dist = RRT(
        start, goals, buffered_obstacles, DIM, NUM_SAMPLES, ax)
    stop_time = timeit.default_timer()

    print("Elapsed time:", stop_time - start_time, len(trajectory))

    for i in range(len(trajectory)):
        if i != len(trajectory) - 1:
            ax.plot([trajectory[i].x, trajectory[i+1].x],
                    [trajectory[i].y, trajectory[i+1].y], "g-", zorder=20)

    plt.show()
