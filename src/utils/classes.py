from dataclasses import dataclass
from typing import Literal
from matplotlib import axes
from matplotlib.patches import Arc
from shapely import Polygon, LineString
import math
import numpy as np


class Point:
    def __init__(self, x: float, y: float, theta: float = 0):
        self.x = x
        self.y = y
        self.theta = theta

    def tuple(self):
        return self.x, self.y


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
        ax.plot([self.start_config.x, self.end_config.x],
                [self.start_config.y, self.end_config.y], color=color, zorder=zorder, linewidth=1)

    def sample_points(self, num_samples, ax: axes.Axes = None):
        xs = np.linspace(self.start_config.x, self.end_config.x, num_samples)
        ys = np.linspace(self.start_config.y, self.end_config.y, num_samples)

        samples = [Point(xs[i], ys[i], self.start_config.theta)
                   for i in range(num_samples)]
                
        if ax is not None:
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

        start_theta_vec = (self.start_config.x - self.center.x,
                           self.start_config.y - self.center.y)
        self.start_theta = math.atan2(start_theta_vec[1], start_theta_vec[0])

    def plot(self, ax: axes.Axes, color="y", zorder=1):
        from src.utils.functions import rad_2_deg

        rot_angle = 0 if self.curve_type == "left" else - \
            rad_2_deg(self.arc_angle)

        arc = Arc(
            self.center.tuple(), height=2*self.radius, width=2*self.radius, facecolor="none", edgecolor=color, zorder=zorder, theta1=rad_2_deg(self.start_theta), theta2=rad_2_deg(self.start_theta)+rad_2_deg(self.arc_angle), angle=rot_angle)

        ax.add_patch(arc)

    # TODO get more accurate bounding box
    def get_bb(self) -> Polygon:
        c = self.center
        r = self.radius

        x_min, x_max = c.x - r,  c.x + r
        y_min, y_max = c.y - r,  c.y + r

        return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

    def sample_points(self, num_samples, ax: axes.Axes = None):
        if self.curve_type == 'left':
            range = np.linspace(
                self.start_theta, self.start_theta + self.arc_angle, num_samples)
        else:
            range = np.linspace(
                self.start_theta, self.start_theta - self.arc_angle, num_samples)

        samples = []

        for th in range:
            x = self.radius * math.cos(th) + self.center.x
            y = self.radius * math.sin(th) + self.center.y

            samples.append(Point(x, y, th))
            if ax is not None:
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
            (self.line.start_config.tuple(), self.line.end_config.tuple()))
        if line_string.intersects(obstacle):
            return True

        if self.curve2 is not None and self.curve2.get_bb().intersects(obstacle):
            return True

        return False

    def sample_points(self, num_path_samples: tuple, ax: axes.Axes = None):
        samples = self.curve1.sample_points(num_path_samples[0], ax)
        samples.extend(self.line.sample_points(num_path_samples[1], ax))

        if self.curve2 is not None:
            samples.extend(self.curve2.sample_points(num_path_samples[0], ax))

        return samples
