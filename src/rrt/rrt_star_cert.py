from matplotlib import axes, pyplot as plt
from scipy.spatial import KDTree
from shapely import LineString, distance, Polygon, Point as ShapelyPoint
import numpy as np
import math
import timeit
from src.utils import setup_rrt_plot, euc_distance, Point, create_halton_sample
from typing import List


def check_line_collision(obstacles: List[Polygon], point1: Point, point2: Point):
    is_collision = False

    for obstacle in obstacles:
        if obstacle.intersects(LineString((point1.tuple(), point2.tuple()))):
            is_collision = True
            break

    return is_collision


def create_certificate(point: Point, obstacles: list):
    min_dist = math.inf

    for obstacle in obstacles:
        dist = distance(ShapelyPoint(point.tuple()), obstacle)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def RRT(start: Point, goal: Point, obstacles, dim, num_samples, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]

    certificates_ref = {0: 0}
    certificates = {0: create_certificate(start, obstacles)}

    for sample in range(num_samples):
        kdtree = KDTree([p.tuple() for p in points])

        new_pt = Point(*create_halton_sample(sample, dim), 0)

        proximity = (math.log(len(points)) / len(points)) * \
            (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2
        if proximity == 0:
            proximity = (dim[1][0] - dim[0][0]) * (dim[1][1] - dim[0][1]) / 2

        nearest_pt_idxs: list = kdtree.query_ball_point(new_pt.tuple(), proximity)

        potential_parents = []
        potential_distances = []
        potential_certificates = []
        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt: Point = points[nearest_pt_idx]

            dist = euc_distance(nearest_pt, new_pt)
            nearest_pt_distances.append(dist)

            new_dist = distances[nearest_pt_idx] + dist

            nearest_cert_idx = certificates_ref[nearest_pt_idx]
            nearest_cert_pt = points[nearest_cert_idx]
            dist_to_cert = euc_distance(nearest_cert_pt, new_pt)

            if dist_to_cert < certificates[nearest_cert_idx]:
                potential_parents.append(nearest_pt_idx)
                potential_distances.append(new_dist)
                potential_certificates.append((nearest_cert_idx, None))

            elif not check_line_collision(obstacles, nearest_pt, new_pt):
                potential_parents.append(nearest_pt_idx)
                potential_distances.append(new_dist)
                potential_certificates.append(
                    (None, create_certificate(new_pt, obstacles)))

        if len(potential_distances) == 0:
            continue

        min_idx = np.argmin(potential_distances)
        new_dist = potential_distances[min_idx]
        new_parent_idx = potential_parents[min_idx]

        points.append(new_pt)
        parents.append(new_parent_idx)
        distances.append(new_dist)

        new_pt_idx = len(points) - 1

        if potential_certificates[min_idx][0] == None:
            certificates_ref[new_pt_idx] = new_pt_idx
        else:
            certificates_ref[new_pt_idx] = potential_certificates[min_idx][0]

        if potential_certificates[min_idx][1] != None:
            certificates[new_pt_idx] = potential_certificates[min_idx][1]

        for i, remaining_pt_idx in enumerate(nearest_pt_idxs):
            if remaining_pt_idx == new_parent_idx:
                continue

            remaining_pt = points[remaining_pt_idx]
            nearest_to_new_dist = nearest_pt_distances[i]

            if new_dist + nearest_to_new_dist < distances[remaining_pt_idx]:

                nearest_cert_idx = certificates_ref[new_pt_idx]
                nearest_cert_pt = points[nearest_cert_idx]
                dist_to_cert = euc_distance(nearest_cert_pt, remaining_pt)

                if dist_to_cert < certificates[nearest_cert_idx]:
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist
                    certificates_ref[remaining_pt_idx] = nearest_cert_idx

                elif not check_line_collision(obstacles, remaining_pt, new_pt):
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

                    certificates_ref[remaining_pt_idx] = remaining_pt_idx
                    certificates[remaining_pt_idx] = create_certificate(
                        remaining_pt, obstacles)

        ax.plot([new_pt.x, points[potential_parents[min_idx]].x], [
            new_pt.y, points[potential_parents[min_idx]].y], "y-")

    kdtree = KDTree([p.tuple() for p in points])
    dist, nearest_pt_idx = kdtree.query(goal.tuple())
    nearest = points[nearest_pt_idx]

    trajectory: List[Point] = [goal, nearest]
    total_dist = distances[nearest_pt_idx] + dist

    while nearest != start:
        nearest_pt_idx = parents[nearest_pt_idx]
        parent = points[nearest_pt_idx]

        trajectory.append(parent)

        nearest = parent

    return trajectory, total_dist


if __name__ == "__main__":

    start = Point(1, 2, 0)
    goal = Point(8, 9, 0)

    NUM_SAMPLES = 1000
    DIM = ((0, 0), (10, 10))
    VEHICLE_RADIUS = 0.3

    obstacles = [
        Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        Polygon(((6, 6), (7, 5), (8, 6), (8, 8), (6, 8))),
        Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3)))
    ]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            VEHICLE_RADIUS, join_style="mitre"))

    ax = setup_rrt_plot(DIM, start, goal, obstacles,
                        buffered_obstacles, VEHICLE_RADIUS)

    start_time = timeit.default_timer()
    trajectory, dist = RRT(
        start, goal, buffered_obstacles, DIM, NUM_SAMPLES, ax)
    stop_time = timeit.default_timer()

    print("Elapsed time:", stop_time - start_time)

    for i in range(len(trajectory)):
        if i != len(trajectory) - 1:
            ax.plot([trajectory[i].x, trajectory[i+1].x],
                    [trajectory[i].y, trajectory[i+1].y], "g-", zorder=10)

    plt.show()
