from matplotlib import pyplot as plt, axes
from scipy.spatial import KDTree
from shapely import Polygon
import numpy as np
import math
from typing import List
from src.utils import setup_rrt_plot, create_halton_sample, euc_distance, mod_pi, rad_2_deg, mod_2_pi
from src.classes import Point, Path
from src.dubins.dubins_circle_to_point import calculate_dubins_path
from src.pid import PIDController
from src.vehicle_dynamics import diff_dynamics


def check_path_collision(obstacles, path: Path):
    is_collision = False

    for obstacle in obstacles:
        if path.intersects(obstacle):
            is_collision = True
            break

    return is_collision


def RRT_step(points, distances, parents, dim, index, obstacles, dubins_radius, ax):
    kdtree = KDTree([p.tuple() for p in points])

    new_pt_coords = create_halton_sample(index, dim)

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

    if len(potential_paths) != 0:

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

    return points, distances, parents


def get_RRT_path(points, distances, parents, goal, dubins_radius, obstacles):
    kdtree = KDTree([p.tuple() for p in points])
    start = points[0]

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


def RRT(start: Point, goal: Point, obstacles: List[Polygon], dim, num_samples, dubins_radius, ax: axes.Axes):
    points: List[Point] = [start]
    parents = [None]
    distances = [0]

    for sample in range(num_samples):
        RRT_step(points, distances, parents, dim,
                 sample, obstacles, dubins_radius, ax)

    return get_RRT_path(points, distances, parents, goal, dubins_radius, obstacles)


def follow_path():
    start = Point(0, 9, 1)
    goal = Point(4.5, -2)

    NUM_SAMPLES = 40
    DIM = ((-11, -11), (11, 11))
    VEHICLE_RADIUS = 0.3
    DUBINS_RADIUS = 0.45
    NUM_PATH_SAMPLES = (10, 10)
    WHEELBASE = 1

    obstacles = [
        # Polygon(((1, 4), (3, 4), (4, 6), (2, 7), (1, 6))),
        # Polygon(((4, 7), (6, 7), (6, 9), (4, 9), (3, 8))),
        # Polygon(((4, 2), (6, 2), (6, 4), (4, 4), (3, 3))),


        Polygon(([-7, -7], [-7, -3], [-3, -3], [-3, -7], [-7, -7])),
        Polygon(([-1, -7], [-1, -3], [3, -3], [3, -7], [-1, -7])),
        Polygon(([5,  -7], [5, -3], [9, -3], [9, -7], [5, -7])),
        Polygon(([-7, -1], [-7, 3], [-3, 3], [-3, -1], [-7, -1])),
        Polygon(([-1, -1], [-1, 3], [3, 3], [3, -1], [-1, -1])),
        Polygon(([5, -1], [5, 3], [9, 3], [9, -1], [5, -1])),
        Polygon(([-6, 5], [-5, 6], [-4, 6], [-5, 4], [-6, 5])),
        Polygon(([0, 5], [1, 7], [2, 6], [1, 4], [0, 5])),
        Polygon(([6, 5], [7, 7], [8, 6], [7, 4], [6, 5])),
    ]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            VEHICLE_RADIUS, join_style="mitre"))

    pid = PIDController(0.1, 0, 1)

    desired_x = []
    desired_y = []
    desired_theta = []
    actual_x = []
    actual_y = []
    actual_theta = []
    current_idx = []

    current_x = start.x
    current_y = start.y
    current_th = start.theta

    w = 0

    ax0: axes.Axes
    ax1: axes.Axes
    ax2: axes.Axes
    ax3: axes.Axes
    ax4: axes.Axes
    fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(5, 1)

    nearest_idx = 0

    rrt_index = 0
    rrt_points = []
    rrt_distances = []
    rrt_parents = []
    rrt_path = []

    for _ in range(10000):

        current = Point(current_x, current_y, current_th)
        ax4.plot(current_x, current_y, "bo", zorder=100)

        if rrt_index == 0:
            rrt_points = [current]
            rrt_parents = [None]
            rrt_distances = [0]
            ax4.clear()
            setup_rrt_plot(DIM, start, goal, obstacles,
                           buffered_obstacles, VEHICLE_RADIUS, ax4)

        if rrt_index < NUM_SAMPLES:
            rrt_points, rrt_distances, rrt_parents = RRT_step(
                rrt_points,
                rrt_distances,
                rrt_parents,
                DIM,
                rrt_index,
                buffered_obstacles,
                DUBINS_RADIUS,
                ax4
            )
            rrt_index += 1

        if rrt_index == NUM_SAMPLES:
            dubins_paths, _ = get_RRT_path(
                rrt_points,
                rrt_distances,
                rrt_parents,
                goal,
                DUBINS_RADIUS,
                buffered_obstacles,
            )

            rrt_path = []

            for path in dubins_paths:
                path.plot(ax4, color="g", zorder=10)
                ax4.plot(path.curve1.end_config.x,
                         path.curve1.end_config.y, "go", zorder=30)
                ax4.plot(path.line.end_config.x,
                         path.line.end_config.y, "go", zorder=30)

                rrt_path[:0] = path.sample_points(ax4, NUM_PATH_SAMPLES)

            if len(rrt_path) > 0:
                kdtree = KDTree([p.tuple() for p in rrt_path])
                _, nearest_idx = kdtree.query((current_x, current_y))

            rrt_index = 0
            # rrt_index += 1

        if len(rrt_path) != 0:
            next_point = rrt_path[nearest_idx]
            dist = euc_distance(current, next_point)

            while dist < 0.1 and nearest_idx < len(rrt_path) - 1:
                nearest_idx += 1
                next_point = rrt_path[nearest_idx]
                dist = euc_distance(current, next_point)

            theta_desired = math.atan2(
                (next_point.y - current.y), (next_point.x - current.x)
            )

            diff_theta = mod_pi(theta_desired - current.theta)
            acc_theta = pid.calculate(diff_theta, 0)

            v = (1 - math.e ** (- 1 * dist)) * math.e ** (- abs(diff_theta))
            w += acc_theta

            if w > 5:
                w = 5
            if w < -5:
                w = -5

            dx, dy, dth = diff_dynamics(v, current_th, w, WHEELBASE)

            print(nearest_idx, len(rrt_path), "\n", current_x, next_point.x, "\n", current_y, next_point.y, "\n",
                  rad_2_deg(mod_2_pi(current_th)), rad_2_deg(mod_2_pi(theta_desired)), rad_2_deg(diff_theta), "\n", v, w, "\n", dx, dy, dth)
            print("------")

            current_x += dx
            current_y += dy
            current_th += dth

            desired_x.append(next_point.x)
            desired_y.append(next_point.y)
            desired_theta.append(rad_2_deg(mod_2_pi(theta_desired)))
            actual_x.append(current.x)
            actual_y.append(current.y)
            actual_theta.append(rad_2_deg(mod_2_pi(current.theta)))
            current_idx.append(nearest_idx)

            ax0.clear()
            ax0.plot(current_idx, "b")

            ax1.clear()
            ax1.plot(desired_x, "b")
            ax1.plot(actual_x, "r")

            ax2.clear()
            ax2.plot(desired_y, "b")
            ax2.plot(actual_y, "r")

            ax3.clear()
            ax3.set_ylim(0, 360)
            ax3.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
            ax3.plot(desired_theta, "b")
            ax3.plot(actual_theta, "r")

            plt.draw()
            plt.waitforbuttonpress()

    plt.show()


if __name__ == "__main__":
    follow_path()
