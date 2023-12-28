from matplotlib import pyplot as plt, axes
from scipy.spatial import KDTree
from shapely import LineString, Polygon
import numpy as np
import math
from typing import List
from src.utils import Point, deg_2_rad, Path
import timeit
from src.dubins.dubins_circle_to_point import calculate_dubins_path

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

def check_line_collision(obstacles, point1, point2):
    is_collision = False

    for obstacle in obstacles:
        if obstacle.intersects(LineString((point1, point2))):
            is_collision = True
            break

    return is_collision


def euc_dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2) ** (1 / 2)

        
def RRT(init: Point, goal: Point, obstacles: List[Polygon], dim, num_samples, vehicle_radius, dubins_radius, ax: axes.Axes):
    points: List[Point] = [init]
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

        nearest_pt_idxs: List[int] = kdtree.query_ball_point(new_pt_coords, proximity)

        potential_parents = []
        potential_paths: List[Path] = []
        potential_distances = []
        
        nearest_pt_distances = []

        for nearest_pt_idx in nearest_pt_idxs:
            nearest_pt = points[nearest_pt_idx]

            ppp = Point(new_pt_coords, 0)
            dubins_path = calculate_dubins_path(nearest_pt, ppp, dubins_radius)

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

        
        ppp = dubins_path.line.end_config

        points.append(ppp)
        parents.append(new_parent_idx)
        distances.append(new_dist)

        new_pt_idx = len(points) - 1

        for i, remaining_pt_idx in enumerate(nearest_pt_idxs):
            if remaining_pt_idx == new_parent_idx:
                continue
            
            remaining_pt = points[remaining_pt_idx]

            remaining_dubins_path: Path = calculate_dubins_path(ppp, remaining_pt, dubins_radius)
            nearest_to_new_dist = remaining_dubins_path.length
            
            if new_dist + nearest_to_new_dist < distances[remaining_pt_idx]:
                if not check_path_collision(buffered_obstacles, remaining_dubins_path):
                    parents[remaining_pt_idx] = new_pt_idx
                    distances[remaining_pt_idx] = new_dist + \
                        nearest_to_new_dist

        # ax.plot([new_pt[0], points[potential_parents[min_idx]][0]], [
        #     new_pt[1], points[potential_parents[min_idx]][1]], "y-")

        dubins_path.ax = ax
        dubins_path.plot()
        # plt.pause(PLOT_DELAY)
        
    kdtree = KDTree([p.p for p in points])
    dist, nearest_pt_idx = kdtree.query(goal.p)
    nearest = points[nearest_pt_idx]

    total_dist = distances[nearest_pt_idx] + dist
    
    trajectory = [calculate_dubins_path(nearest, goal, dubins_radius)]
    
    while nearest != init:
        nearest_pt_idx = parents[nearest_pt_idx]
        parent = points[nearest_pt_idx]

        trajectory.append(calculate_dubins_path(parent, nearest, dubins_radius))

        nearest = parent

    return trajectory, total_dist


def setup_plot(dim, init: Point, goal: Point, obstacles, vehicle_radius):
    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, dim[0])
    ax.set_ylim(0, dim[1])
    ax.set_aspect('equal')

    init_circle = plt.Circle(init.p, vehicle_radius,
                             facecolor='none', edgecolor='b', zorder=15)
    ax.add_patch(init_circle)
    ax.plot(init.p[0], init.p[1], "bo", zorder=15)
    ax.plot(goal.p[0], goal.p[1], "go", zorder=15)

    for o in obstacles:
        x, y = o.exterior.xy
        ax.plot(x, y, "r")
        x, y = o.buffer(vehicle_radius, join_style="mitre").exterior.xy
        ax.plot(x, y, color="orange")

    return ax


if __name__ == "__main__":

    init = Point((1, 2), deg_2_rad(-90))
    goal = Point((8, 9), 0)

    obstacles = [
        Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        Polygon(((6, 6), (7, 5), (8, 6), (8, 8), (6, 8))),
        Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3)))
    ]

    NUM_SAMPLES = 1000
    DIM = (10, 10)
    VEHICLE_RADIUS = 0.3
    PLOT_DELAY = 0.3
    DUBINS_RADIUS = 0.1

    ax = setup_plot(DIM, init, goal, obstacles, VEHICLE_RADIUS)

    start = timeit.default_timer()
    traj, dist = RRT(init, goal, obstacles, DIM,
                        NUM_SAMPLES, VEHICLE_RADIUS, DUBINS_RADIUS, ax)
    stop = timeit.default_timer()

    print("Elapsed time:", stop - start)

    for path in traj:
        path.ax = ax
        path.plot(color="g", zorder=10)

    plt.show()
