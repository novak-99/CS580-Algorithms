# Algorithm from lecture 20

import numpy as np

# init the rng
rng = np.random.default_rng()

def particle_filter(X_prev, u, z, process_noise=0.25, measurement_noise=0.05, M=100):
    X = []

    for m in range(M):
        # Assuming basic 1D motion model
        x = np.random.normal(X_prev[m][0] + u, process_noise) 
        w = np.random.normal(z, measurement_noise)
        X.append((x, w))

    x = np.array([xi for xi, _ in X])
    w = np.array([wi for _, wi in X])

    # Normalize
    w = w / np.sum(w)

    # Resample.
    for m in range(M): X[m] = (float(rng.choice(x, p=w)), float(w[m]))

    return X


def average_particle(X):
    x = np.array([xi for xi, _ in X])
    w = np.array([wi for _, wi in X])

    return (x @ w) / np.sum(w)


true_x = 0.0

u = 1.0

T = 40 

M = 100

X = []
for i in range(M): X.append((true_x, 1.0/M))

for t in range(T):
    true_x += u + np.random.normal(0, 0.25)
    z = true_x + np.random.normal(0, 0.05)

    X = particle_filter(X, u, z, M=M)

    print(true_x)
    print(average_particle(X))