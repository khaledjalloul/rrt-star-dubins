import math
from src.utils import Point


def get_wheel_vel(v, w, l):
    omega_l = v - w * (l / 2)
    omega_r = v + w * (l / 2)

    return omega_l, omega_r


def get_velocities(current: Point, next: Point, l: float):
    dy = next.p[1] - current.p[1]
    dx = next.p[0] - current.p[0]
    d_theta = next.theta - current.theta

    v = (dx ** 2 + dy ** 2) ** (1 / 2)
    w = d_theta * l / v
    w = math.atan(w)

    return v, w
