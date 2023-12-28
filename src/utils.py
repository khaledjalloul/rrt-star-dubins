
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
class Line:
    start_config: Point
    end_config: Point
    length: float = 0


@dataclass
class Circle:
    radius: float
    center: Point
    curve_type: Literal["left", "right"]


@dataclass
class Curve:
    start_config: Point
    end_config: Point
    center: Point
    radius: float
    curve_type: Literal["left", "right"]
    arc_angle: float
    length: float


class Path:
    def __init__(self, curve1: Curve, line: Line, curve2: Curve = None):
        self.curve1 = curve1
        self.line = line
        self.curve2 = curve2
        self.ax: axes.Axes = None

        self.length = curve1.length + line.length
        self.length += 0 if curve2 is None else curve2.length

    def plot_curve(self, curve: Curve, color="y", zorder=1):
        arc_theta_vec = (curve.start_config.p[0] - curve.center.p[0],
                         curve.start_config.p[1] - curve.center.p[1])
        arc_theta = math.atan2(arc_theta_vec[1], arc_theta_vec[0])

        arc_angle = 0 if curve.curve_type == "left" else - \
            rad_2_deg(curve.arc_angle)

        arc = Arc(
            curve.center.p, height=2*curve.radius, width=2*curve.radius, facecolor="none", edgecolor=color, zorder=1, theta1=rad_2_deg(arc_theta), theta2=rad_2_deg(arc_theta)+rad_2_deg(curve.arc_angle), angle=arc_angle)

        self.ax.add_patch(arc)

    def plot(self, color="y", zorder=1):
        self.plot_curve(self.curve1, color, zorder)

        self.ax.plot([self.line.start_config.p[0], self.line.end_config.p[0]],
                [self.line.start_config.p[1], self.line.end_config.p[1]], color=color, zorder=zorder, linewidth=1)

        if self.curve2 is not None:
            self.plot_curve(self.curve2)

    # TODO
    def curve_bb_todo(self, curve: Curve) -> Polygon:
        ps = np.r_[[curve.start_config.p],
                   [curve.end_config.p]]

        x_min, y_min = np.min(ps, axis=0)
        x_max, y_max = np.max(ps, axis=0)

        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min)
        self.ax.add_patch(rect)
        return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    def curve_bb(self, curve: Curve) -> Polygon:
        c = curve.center.p
        r = curve.radius

        x_min, x_max = c[0] - r,  c[0] + r
        y_min, y_max = c[1] - r,  c[1] + r

        return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    def intersects(self, obstacle: Polygon):
        if self.curve_bb(self.curve1).intersects(obstacle):
            return True

        line_string = LineString(
            (self.line.start_config.p, self.line.end_config.p))
        if line_string.intersects(obstacle):
            return True

        if self.curve2 is not None and self.curve_bb(self.curve2).intersects(obstacle):
            return True

        return False


def mod_2_pi(x: float) -> float:
    return x - 2 * np.pi * np.floor(x / (2 * np.pi))


def rad_2_deg(angle: float) -> float:
    return angle * 180 / math.pi


def deg_2_rad(angle: float) -> float:
    return angle * math.pi / 180


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