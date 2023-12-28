from matplotlib import axes, pyplot as plt
from scipy.spatial import KDTree
from shapely import LineString, distance, Point, Polygon
import numpy as np
import math
import timeit
from src.utils import setup_rrt_plot

def create_sample(i, dim):
    x = np.base_repr(i, base=2)[::-1]
    y = np.base_repr(i, base=3)[::-1]

    x_new, y_new = 0, 0

    for i, d in enumerate(x):
        x_new += int(d) / (2 ** (i+1))
    for i, d in enumerate(y):
        y_new += int(d) / (3 ** (i+1))

    return x_new * dim[0], y_new * dim[1]


def check_line_collision(obstacles, point1, point2):
    is_collision = False

    for obstacle in obstacles:
        if obstacle.intersects(LineString((point1, point2))):
            is_collision = True
            break

    return is_collision


def create_certificate(point, obstacles: list):
    min_dist = math.inf

    for obstacle in obstacles:
        dist = distance(Point(*point), obstacle)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def euc_dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2) ** (1 / 2)


def RRT(init, goal, obstacles, dim, num_samples, vehicle_radius, ax: axes.Axes):
    points = [init]
    parents = [None]
    distances = [0]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            vehicle_radius, join_style="mitre"))

    certificates_ref = {0: 0}
    certificates = {0: create_certificate(init, buffered_obstacles)}

    for sample in range(num_samples):
        kdtree = KDTree(points)

        new_pt = create_sample(sample, dim)

        proximity = (math.log(len(points)) / len(points)) * dim[0] * dim[1] / 2
        if proximity == 0:
            proximity = dim[0] * dim[1] / 2

        nearest_pt_idxs: list = kdtree.query_ball_point(new_pt, proximity)

        potential_parents = []
        potential_distances = []
        potential_certificates = []
        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt = points[nearest_pt_idx]

            dist = euc_dist(nearest_pt, new_pt)
            nearest_pt_distances.append(dist)

            new_dist = distances[nearest_pt_idx] + dist

            nearest_cert_idx = certificates_ref[nearest_pt_idx]
            nearest_cert_pt = points[nearest_cert_idx]
            dist_to_cert = euc_dist(nearest_cert_pt, new_pt)

            if dist_to_cert < certificates[nearest_cert_idx]:
                potential_parents.append(nearest_pt_idx)
                potential_distances.append(new_dist)
                potential_certificates.append((nearest_cert_idx, None))

            elif not check_line_collision(buffered_obstacles, nearest_pt, new_pt):
                potential_parents.append(nearest_pt_idx)
                potential_distances.append(new_dist)
                potential_certificates.append(
                    (None, create_certificate(new_pt, buffered_obstacles)))

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
                dist_to_cert = euc_dist(nearest_cert_pt, remaining_pt)

                if dist_to_cert < certificates[nearest_cert_idx]:
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist
                    certificates_ref[remaining_pt_idx] = nearest_cert_idx

                elif not check_line_collision(buffered_obstacles, remaining_pt, new_pt):
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

                    certificates_ref[remaining_pt_idx] = remaining_pt_idx
                    certificates[remaining_pt_idx] = create_certificate(
                        remaining_pt, buffered_obstacles)

        ax.plot([new_pt[0], points[potential_parents[min_idx]][0]], [
            new_pt[1], points[potential_parents[min_idx]][1]], "y-")

    kdtree = KDTree(points)
    dist, nearest_pt_idx = kdtree.query(goal)
    nearest = points[nearest_pt_idx]

    trajectory = [goal, nearest]
    total_dist = distances[nearest_pt_idx] + dist

    while nearest != init:
        nearest_pt_idx = parents[nearest_pt_idx]
        parent = points[nearest_pt_idx]

        trajectory.append(parent)

        nearest = parent

    return trajectory, total_dist


if __name__ == "__main__":

    init = (1, 2)
    goal = (8, 9)

    obstacles = [
        Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        Polygon(((6, 6), (7, 5), (8, 6), (8, 8), (6, 8))),
        Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3)))
    ]

    NUM_SAMPLES = 1000
    DIM = (10, 10)
    VEHICLE_RADIUS = 0.3

    ax = setup_rrt_plot(DIM, init, goal, obstacles, VEHICLE_RADIUS)

    start = timeit.default_timer()
    traj, dist = RRT(init, goal, obstacles, DIM,
                        NUM_SAMPLES, VEHICLE_RADIUS, ax)
    stop = timeit.default_timer()

    print("Elapsed time:", stop - start)

    for i in range(len(traj)):
        if i != len(traj) - 1:
            ax.plot([traj[i][0], traj[i+1][0]],
                    [traj[i][1], traj[i+1][1]], "g-", zorder=10)

    plt.show()
