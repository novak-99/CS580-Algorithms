# algorithms from lecture 13

import numpy as np

def linear_motion(theta_start, theta_end, s, t):
    return theta_start + s(t) * (theta_end - theta_start)

def cubic_time_scaling(a0, a1, a2, a3, t):
    return a0 + a1 * t + a2 * t**2 + a3 * t**3

# s(0) = 0, s(T) = 1
# s_dot(0) = 0, s_dot(T) = 0
# forces the following.
def cubic_time_scaling_rest(t, T):
    return cubic_time_scaling(0, 0, (3*t**2)/T**2, (2*t**3)/T**3)

def theta_vel_max(T, theta_start, theta_end):
    return 3/(2*T) * (theta_end - theta_start)

def theta_acc_max(T, theta_start, theta_end):
    return np.abs(6/T**2 * (theta_end - theta_start))

def trapezoidal_scaling(t, T, v, a):
    if t >= 0 and t <= v/a:
        return 0.5*a*t**2
    elif t > v/a and t <= T - v/a:
        return v*t - v**2 / 2*a
    else:
        return (2*a*v*T - 2*v**2 - a**2*(t - T)**2)/(2*a)