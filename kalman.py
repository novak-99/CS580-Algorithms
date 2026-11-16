# Algorithms from lecture 19
import numpy as np

def kalman_filter(mu_prev, Sigma_prev, u, z, A, B, C, R, Q):
    # Prediction update
    mu_pred = A @ mu_prev + B @ u
    Sigma_pred = A @ Sigma_prev @ A.T + R

    # Kalman gain 
    # (~ how much we should trust measurement)
    K = Sigma_pred @ C.T @ np.linalg.inv(C @ Sigma_prev @ C.T + Q)

    # Measurement update
    # (~ how much should we use measurement err to update state)
    mu = mu_pred + K @ (z - C @ mu_pred)
    Sigma = Sigma_pred - K @ C @ Sigma_pred

    return (mu, Sigma)

# Same as kalman, except G and H replace A and C,
# and you evaluate g(*) and h(*) instead of using 
# the Linear transform
def extended_kalman_filter(mu_prev, Sigma_prev, u, z, G, H, R, Q, g, h):
    # Prediction update
    mu_pred = g(u, mu_prev)
    Sigma_pred = G @ Sigma_prev @ G + R

    # Kalman gain
    K = Sigma_pred @ H.T @ np.linalg.inv(H @ Sigma_pred @ H.T + Q)

    # Measurement update
    mu = mu_pred + K @ (z - h(mu_pred))
    Sigma = Sigma_prev - K @ H @ Sigma_prev

    return (mu, Sigma)

T = 1000
dt = 1 # advance 1 second every epoch

A = np.array(
        [[1, dt],
        [0, 1]]
    )

B = np.array(
    [[0.5*dt**2], [dt]]
)

# no acceleration
a = np.array([[0]])

# Sensor only measures position
C = np.array(
    [[1, 0]]
)

# Process Nosie
R = np.array(
        [[0.01, 0],
         [0, 0.01]]
    )

# Measurement noise
Q = np.array([[0.25]])

# Initial true state, mean and variance
x_initial = np.array([0, 1])

x_true = x_initial
mu = x_initial
Sigma = np.eye(2)

for t in range(T):
    x_true = A @ x_true
    z = x_true + np.random.multivariate_normal(np.zeros(2), R)

    mu, Sigma = kalman_filter(mu, Sigma, a, z, A, B, C, R, Q)

    print(x_true)
    print(mu, Sigma) 