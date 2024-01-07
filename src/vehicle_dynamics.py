import math


def get_wheel_vel(v, w, l):
    omega_l = v - w * (l / 2)
    omega_r = v + w * (l / 2)

    return omega_l, omega_r


def diff_dynamics(v, theta, w, L):
    d_x = v * math.cos(theta)
    d_y = v * math.sin(theta)
    d_theta = (v / L) * math.tan(w)
    d_theta = w

    return d_x, d_y, d_theta
