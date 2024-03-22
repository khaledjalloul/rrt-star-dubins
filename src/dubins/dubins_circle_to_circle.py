
import math
from typing import List
from matplotlib import pyplot as plt, axes
from src.utils.functions import mod_2_pi, deg_2_rad
from src.utils.classes import *


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


def calculate_tangent_btw_circles(circle_start: Circle, circle_end: Circle) -> List[Line]:
    C1 = circle_start.center
    C2 = circle_end.center
    R = circle_start.radius
    direction1 = circle_start.curve_type
    direction2 = circle_end.curve_type

    tangents = []

    C1C2 = round(math.sqrt((C1.x - C2.x) ** 2 + (C1.y - C2.y) ** 2), 8)
    O = ((C1.x + C2.x) / 2, (C1.y + C2.y) / 2)

    vec = (C2.x - C1.x, C2.y - C1.y)

    points = []

    if direction1 == direction2:
        if direction1 == "right":
            # External tangent 1 (to the left)
            vec_perp = (-vec[1], vec[0])

        if direction1 == "left":
            # External tangent 1 (to the right)
            vec_perp = (vec[1], -vec[0])

        vec_perp = (vec_perp[0] / C1C2 * R, vec_perp[1] / C1C2 * R)

        p1 = (C1.x + vec_perp[0], C1.y + vec_perp[1])
        p2 = (C2.x + vec_perp[0], C2.y + vec_perp[1])
        theta = math.atan2(vec[1], vec[0])

        points.append(Line(Point(*p1, theta), Point(*p2, theta)))

    if direction1 != direction2 and C1C2 >= 2 * R:

        if C2.x - C1.x == 0:
            slope_C1C2 = float("inf")
        else:
            slope_C1C2 = (C2.y - C1.y) / (C2.x - C1.x)

        angle_tangent_C1C2 = math.asin(R / (C1C2 / 2))
        angle_C1C2 = math.atan(slope_C1C2)

        if direction1 == "right" and direction2 == "left":
            # Internal tangent 1
            angle_tangent3 = angle_C1C2 - angle_tangent_C1C2
            slope_tangent3 = math.tan(angle_tangent3)
            if (abs(slope_tangent3) < 99999):
                y_intercept_tangent3 = O[1] - slope_tangent3 * O[0]
                tangents.append((slope_tangent3, y_intercept_tangent3, None))
            else:
                x_intercept_tangent3 = O[0]
                tangents.append((None, None, x_intercept_tangent3))

        if direction1 == "left" and direction2 == "right":
            # Internal tangent 2
            angle_tangent4 = angle_C1C2 + angle_tangent_C1C2
            slope_tangent4 = math.tan(angle_tangent4)
            if (abs(slope_tangent4) < 99999):
                y_intercept_tangent4 = O[1] - slope_tangent4 * O[0]
                tangents.append((slope_tangent4, y_intercept_tangent4, None))
            else:
                x_intercept_tangent4 = O[0]
                tangents.append((None, None, x_intercept_tangent4))

    for tangent in tangents:
        if tangent[2] != None:
            theta = math.atan2(C2.y - C2.x, 0)
            points.append(Line(Point(tangent[2], C1.y, theta),
                          Point(tangent[2], C2.y, theta)))
        elif tangent[0] == 0:
            theta = math.atan2(0, C2.x - C1.x)
            points.append(
                Line(Point(C1.x, tangent[1], theta), Point(C2.x, tangent[1], theta)))
        else:
            slope_perpendicular = - 1 / tangent[0]

            y_intercept_perpendicular1 = C1.y - slope_perpendicular * C1.x
            x1 = (y_intercept_perpendicular1 -
                  tangent[1]) / (tangent[0] - slope_perpendicular)
            y1 = tangent[0] * x1 + tangent[1]

            y_intercept_perpendicular2 = C2.y - slope_perpendicular * C2.x
            x2 = (y_intercept_perpendicular2 -
                  tangent[1]) / (tangent[0] - slope_perpendicular)
            y2 = tangent[0] * x2 + tangent[1]

            theta = math.atan(tangent[0])
            if (tangent[0] > 0 and y1 > y2) or (tangent[0] < 0 and y1 < y2):
                theta = mod_2_pi(theta + math.pi)

            points.append(Line(Point(x1, y1, theta), Point(x2, y2, theta)))

    return points


def calculate_dubins_path(start_config: Point, end_config: Point, radius: float) -> Path:
    start_circles = calculate_turning_circles(start_config, radius)
    end_circles = calculate_turning_circles(end_config, radius)

    path = []
    length = float("inf")

    for start_circle in (start_circles[0], start_circles[1]):
        for end_circle in (end_circles[0], end_circles[1]):
            tangent = calculate_tangent_btw_circles(start_circle, end_circle)
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

            arc_length2 = round(math.sqrt((end_config.x - tangent[0].end_config.x) ** 2 + (
                end_config.y - tangent[0].end_config.y) ** 2), 8)
            if arc_length2 > 2 * radius:
                continue
            arc_angle2 = math.acos(
                (2 * radius ** 2 - arc_length2 ** 2) / (2 * radius ** 2))

            c2 = (tangent[0].end_config.x - end_circle.center.x) * (end_config.y - end_circle.center.y) \
                - (tangent[0].end_config.y - end_circle.center.y
                   ) * (end_config.x - end_circle.center.x)
            if (c2 > 0 and end_circle.curve_type == "right") or (c2 < 0 and end_circle.curve_type == "left"):
                arc_angle2 = 2 * math.pi - arc_angle2

            line_len = ((tangent[0].start_config.x - tangent[0].end_config.x) ** 2 + (
                tangent[0].start_config.y - tangent[0].end_config.y) ** 2) ** (1 / 2)

            temp_path = Path(Curve(start_config, tangent[0].start_config, start_circle.center, radius, start_circle.curve_type, arc_angle1, radius * arc_angle1),
                             Line(tangent[0].start_config,
                                  tangent[0].end_config, line_len),
                             Curve(tangent[0].end_config, end_config, end_circle.center, radius, end_circle.curve_type, arc_angle2, radius * arc_angle2))

            temp_length = temp_path.length
            if temp_length < length:
                path = temp_path
                length = temp_length

    return path


if __name__ == "__main__":
    path = calculate_dubins_path(
        Point(3, 3, deg_2_rad(-90)), Point(8, 8, deg_2_rad(-45)), 1)

    ax: axes.Axes
    fig, ax = plt.subplots()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    path.plot(ax)

    plt.show()
