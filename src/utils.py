
from dataclasses import dataclass
from typing import Tuple, Literal
from matplotlib import axes, pyplot as plt
from matplotlib.patches import Arc, Rectangle
from shapely import Polygon, LineString
import math
import numpy as np


@dataclass
class Point:
    p: Tuple[float, float]
    theta: float = 0


@dataclass
class Circle:
    radius: float
    center: Point
    curve_type: Literal["left", "right"]


@dataclass
class Line:
    def __init__(self, start_config: Point, end_config: Point, length: float = 0):
        self.start_config = start_config
        self.end_config = end_config
        self.length = length

    def plot(self, ax: axes.Axes, color="y", zorder=1):
        ax.plot([self.start_config.p[0], self.end_config.p[0]],
                [self.start_config.p[1], self.end_config.p[1]], color=color, zorder=zorder, linewidth=1)

    def sample_points(self, ax: axes.Axes):
        xs = np.linspace(self.start_config.p[0], self.end_config.p[0], 10)
        ys = np.linspace(self.start_config.p[1], self.end_config.p[1], 10)

        samples = [Point((xs[i], ys[i]), self.start_config.theta)
                   for i in range(10)]
        ax.plot(xs, ys, "ro")

        return samples


@dataclass
class Curve:
    def __init__(self, start_config: Point, end_config: Point, center: Point, radius: float, curve_type: Literal["left", "right"], arc_angle: float, length: float):
        self.start_config = start_config
        self.end_config = end_config
        self.center = center
        self.radius = radius
        self.curve_type = curve_type
        self.arc_angle = arc_angle
        self.length = length

        start_theta_vec = (self.start_config.p[0] - self.center.p[0],
                           self.start_config.p[1] - self.center.p[1])
        self.start_theta = math.atan2(start_theta_vec[1], start_theta_vec[0])

    def plot(self, ax: axes.Axes, color="y", zorder=1):

        rot_angle = 0 if self.curve_type == "left" else - \
            rad_2_deg(self.arc_angle)

        arc = Arc(
            self.center.p, height=2*self.radius, width=2*self.radius, facecolor="none", edgecolor=color, zorder=zorder, theta1=rad_2_deg(self.start_theta), theta2=rad_2_deg(self.start_theta)+rad_2_deg(self.arc_angle), angle=rot_angle)

        ax.add_patch(arc)

    # TODO get more accurate bounding box
    def get_bb(self) -> Polygon:
        c = self.center.p
        r = self.radius

        x_min, x_max = c[0] - r,  c[0] + r
        y_min, y_max = c[1] - r,  c[1] + r

        return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    def sample_points(self, ax: axes.Axes):
        if self.curve_type == 'left':
            range = np.linspace(
                self.start_theta, self.start_theta + self.arc_angle, 10)
        else:
            range = np.linspace(
                self.start_theta, self.start_theta - self.arc_angle, 10)

        samples = []

        for th in range:
            x = self.radius * math.cos(th) + self.center.p[0]
            y = self.radius * math.sin(th) + self.center.p[1]

            samples.append(Point((x, y), th))
            ax.plot(x, y, "ro")

        return samples


class Path:
    def __init__(self, curve1: Curve, line: Line, curve2: Curve = None):
        self.curve1 = curve1
        self.line = line
        self.curve2 = curve2

        self.length = curve1.length + line.length
        self.length += 0 if curve2 is None else curve2.length

    def plot(self, ax: axes.Axes, color="y", zorder=1):
        self.curve1.plot(ax, color, zorder)
        self.line.plot(ax, color, zorder)

        if self.curve2 is not None:
            self.curve2.plot(ax, color, zorder)

    def intersects(self, obstacle: Polygon):
        if self.curve1.get_bb().intersects(obstacle):
            return True

        line_string = LineString(
            (self.line.start_config.p, self.line.end_config.p))
        if line_string.intersects(obstacle):
            return True

        if self.curve2 is not None and self.curve2.get_bb().intersects(obstacle):
            return True

        return False

    def sample_points(self, ax: axes.Axes):
        samples = self.curve1.sample_points(ax)
        samples.extend(self.line.sample_points(ax))

        if self.curve2 is not None:
            samples.extend(self.curve2.sample_points(ax))

        return samples


def mod_2_pi(x: float) -> float:
    return x - 2 * np.pi * np.floor(x / (2 * np.pi))


def rad_2_deg(angle: float) -> float:
    return angle * 180 / math.pi


def deg_2_rad(angle: float) -> float:
    return angle * math.pi / 180


def euc_distance(x: Point, y: Point):
    return ((x.p[0] - y.p[0]) ** 2 + (x.p[1] - y.p[1]) ** 2) ** (1 / 2)


def setup_rrt_plot(dim, init, goal, obstacles, vehicle_radius):
    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, dim[0])
    ax.set_ylim(0, dim[1])
    ax.set_aspect('equal')

    init_circle = plt.Circle(init, vehicle_radius,
                             facecolor='none', edgecolor='b', zorder=15)
    ax.add_patch(init_circle)
    ax.plot(init[0], init[1], "bo", zorder=15)
    ax.plot(goal[0], goal[1], "go", zorder=15)

    for o in obstacles:
        x, y = o.exterior.xy
        ax.plot(x, y, "r")
        x, y = o.buffer(vehicle_radius, join_style="mitre").exterior.xy
        ax.plot(x, y, color="orange")

    return ax
