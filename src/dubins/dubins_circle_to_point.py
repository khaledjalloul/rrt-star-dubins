
import math
import numpy as np
from typing import List
from matplotlib import pyplot as plt, axes
from src.utils import mod_2_pi, deg_2_rad
from src.classes import *


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float):
    min_radius = wheel_base / math.tan(max_steering_angle)
    return min_radius


def calculate_turning_circles(current_config: Point, radius: float):
    theta = mod_2_pi(current_config.theta)
    theta_left = mod_2_pi(theta + math.pi / 2)
    theta_right = mod_2_pi(theta - math.pi / 2)

    left_circle = Circle(center=Point(current_config.x + radius * math.cos(theta_left),
                                      current_config.y + radius * math.sin(theta_left)),
                         radius=radius, curve_type="left")
    right_circle = Circle(center=Point(current_config.x + radius * math.cos(theta_right),
                                       current_config.y + radius * math.sin(theta_right)),
                          radius=radius, curve_type="right")
    return left_circle, right_circle


def calculate_tangent(circle_start: Circle, target: Point) -> List[Line]:
    C1 = circle_start.center
    R = circle_start.radius
    C1P = ((C1.x - target.x) ** 2 + (C1.y - target.y) ** 2) ** (1 / 2)
    direction = circle_start.curve_type

    if R > C1P:
        return []

    tangents = []

    vec = np.array([target.x - C1.x, target.y - C1.y])
    th = math.acos(R / C1P)

    if direction == "right":
        rot_mat = np.array([[math.cos(th), -math.sin(th)],
                           [math.sin(th), math.cos(th)]])

    elif direction == "left":
        rot_mat = np.array([[math.cos(-th), -math.sin(-th)],
                           [math.sin(-th), math.cos(-th)]])

    vec_perp = np.matmul(rot_mat, vec)
    vec_perp = (vec_perp[0] / C1P * R, vec_perp[1] / C1P * R)

    tangent_pt = (C1.x + vec_perp[0], C1.y + vec_perp[1])

    tangent_vec = (target.x - tangent_pt[0], target.y - tangent_pt[1])
    theta = math.atan2(tangent_vec[1], tangent_vec[0])

    tangents.append(Line(Point(*tangent_pt, theta),
                    Point(*target.tuple(), theta)))

    return tangents


def calculate_dubins_path(start_config: Point, end_config: Point, radius: float) -> Path:
    start_circles = calculate_turning_circles(start_config, radius)

    path = []
    length = float("inf")

    for start_circle in (start_circles[0], start_circles[1]):
        tangent = calculate_tangent(start_circle, end_config)
        if len(tangent) == 0:
            continue

        arc_length1 = round(math.sqrt((start_config.x - tangent[0].start_config.x) ** 2 + (
            start_config.y - tangent[0].start_config.y) ** 2), 8)
        if arc_length1 > 2 * radius:
            continue
        arc_angle1 = math.acos(
            (2 * radius ** 2 - arc_length1 ** 2) / (2 * radius ** 2))

        c1 = (start_config.x - start_circle.center.x) * (tangent[0].start_config.y - start_circle.center.y) \
            - (start_config.y - start_circle.center.y) * \
            (tangent[0].start_config.x - start_circle.center.x)
        if (c1 > 0 and start_circle.curve_type == "right") or (c1 < 0 and start_circle.curve_type == "left"):
            arc_angle1 = 2 * math.pi - arc_angle1

        line_len = ((tangent[0].start_config.x - tangent[0].end_config.x) ** 2 + (
            tangent[0].start_config.y - tangent[0].end_config.y) ** 2) ** (1 / 2)

        temp_path = Path(Curve(start_config, tangent[0].start_config, start_circle.center, radius, start_circle.curve_type, arc_angle1, radius * arc_angle1),
                         Line(tangent[0].start_config, tangent[0].end_config, line_len))

        temp_length = temp_path.length
        if temp_length < length:
            path = temp_path
            length = temp_length

    return path


if __name__ == "__main__":

    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    path = calculate_dubins_path(
        Point(3, 3, deg_2_rad(-90)), Point(7, 7, 0), 1)

    path.plot(ax)

    plt.show()
