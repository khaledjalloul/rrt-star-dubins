from matplotlib import pyplot as plt, axes
from shapely import Polygon, LinearRing, Point as ShapelyPoint
import numpy as np
import timeit
from src.utils import setup_rrt_plot, euc_distance
from src.classes import Point
from src.rrt.rrt_star import RRT
from scipy.optimize import minimize
from typing import List


def get_path(ax):

    start = Point(-9, -9, 0)
    goal = ShapelyPoint(9, 9).buffer(0.5)

    NUM_SAMPLES = 400
    DIM = ((-11, -11), (11, 11))
    VEHICLE_RADIUS = 1

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
        Polygon(([-7, -1], [-7, 3], [10, 3], [10, -1], [-7, -1])),
        Polygon(([-1, -1], [-1, 3], [3, 3], [3, -1], [-1, -1])),
        Polygon(([5, -1], [5, 3], [9, 3], [9, -1], [5, -1])),
        Polygon(([0, 5], [1, 7], [2, 6], [1, 4], [0, 5])),
        Polygon(([6, 5], [7, 7], [8, 6], [7, 4], [6, 5])),
        LinearRing(([[-11, -11], [-11, 11], [11, 11], [11, -11], [-11, -11]]))
    ]

    buffered_obstacles = []

    for obstacle in obstacles:
        buffered_obstacles.append(obstacle.buffer(
            VEHICLE_RADIUS, join_style="mitre"))

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

    return trajectory


class RocketOptimizer:
    def __init__(self):
        self.C_T = 0.01
        self.L_M = 0.25
        self.I = 1

        # [x, y, psi, phi, m, dx, dy, dpsi]
        self.X = [-9, -9, 0, 0, 2.5, 0, 0, 0]

        self.phi_limits = (-1.0471975511965976, 1.0471975511965976)
        self.path: List[Point] = []

    def forward_dynamics(self, X, U):

        [x, y, psi, phi, m, dx, dy, dpsi] = X
        [Fl, Fr, dphi] = U

        phi += dphi

        dm = -self.C_T * (Fl + Fr)
        ddpsi = (self.L_M / self.I) * np.cos(phi) * (Fr - Fl)
        ddx = (1 / m) * (np.sin(phi + psi) * Fl + np.sin(phi-psi) * Fr)
        ddy = (1 / m) * (-np.cos(phi + psi) * Fl + np.cos(phi-psi) * Fr)

        dx += ddx
        dy += ddy
        dpsi += ddpsi

        x += dx
        y += dy
        m += dm
        psi += dpsi

        return np.array([x, y, psi, phi, m, dx, dy, dpsi])

    def optim_fun(self, U):
        sum = 0
        X = self.X
        U = np.reshape(U, (-1, 3)).T

        for i, p in enumerate(self.path):
            X = self.forward_dynamics(X, U[:, i])
            sum += np.sum((X[0:3] - [p.x, p.y, p.theta]) ** 2)

        return sum

    def constraint_fun(self, U):
        U = np.reshape(U, (-1, 3)).T
        constraints = []
        X = self.X

        for i, p in enumerate(self.path):
            X = self.forward_dynamics(X, U[:, i])
            constraints.extend([
                X[3] - self.phi_limits[0],
                self.phi_limits[1] - X[3],
                X[4] - 2,
                X[0] - (-11),  # X lower bound
                11 - X[0],  # X upper bound
                X[1] - (-11),  # Y lower bound
                11 - X[1],  # Y upper bound
            ])

        return constraints

    def optim(self, path: List[Point]):
        self.path = path
        self.U = np.tile(np.array([0.0, 0.0, 0.01]), len(path))

        constraints = {'type': 'ineq', 'fun': self.constraint_fun}

        self.U_bounds = [
            (0, 2), (0, 2), (-0.3490658503988659, 0.3490658503988659)] * len(path)

        res = minimize(self.optim_fun,
                       x0=self.U,
                       method="SLSQP",
                       bounds=self.U_bounds,
                       constraints=constraints)

        print("Optimal value:", res.fun)

        self.U = np.reshape(res.x, (-1, 3)).T


if __name__ == "__main__":

    rocket = RocketOptimizer()
    nearest_idx = 0

    ax1: axes.Axes
    ax2: axes.Axes
    _, [ax1, ax2] = plt.subplots(1, 2)

    path1 = get_path(ax1)

    path = []

    for i, p in enumerate(path1):
        if i == len(path1) - 1:
            continue
        xs = np.linspace(p.x, path1[i+1].x, 10)
        ys = np.linspace(p.y, path1[i+1].y, 10)

        theta = np.arctan2(path1[i+1].y - p.y, path1[i+1].x - p.x)

        path.extend([Point(xs[i], ys[i], theta)
                    for i in range(10)])

    ax1.plot([s.x for s in path], [s.y for s in path], "ro")
    ax2.set_aspect('equal')
    ax2.set_xlim(-11, 11)
    ax2.set_ylim(-11, 11)

    rocket.optim(path)
    print(rocket.U.T)
    for u in rocket.U.T:

        # current_p = rocket.p()
        # next_p = path[nearest_idx]
        # dist = euc_distance(current_p, next_p)

        # print(nearest_idx, "\n", rocket.X)
        print(u)
        rocket.X = rocket.forward_dynamics(rocket.X, u)
        ax2.plot(rocket.X[0], rocket.X[1], "bo")
        plt.draw()
        plt.waitforbuttonpress()
        # plt.pause(0.2)

        # while dist < 0.2 and nearest_idx < len(path) - 1:
        #     nearest_idx += 1
        #     next_p = path[nearest_idx]
        #     dist = euc_distance(current_p, next_p)

        # if dist < 0.2 and nearest_idx == len(path) - 1:
        #     break

        # rocket.optim_step(next_p)
        # print("---------")

    plt.show()
