# Algorithm from lecture 12

from alpha_star import a_star, l2_dist
import numpy as np

# init the rng
rng = np.random.default_rng()

# This is the size of our search space;
# 100 ft x 100 ft
C_space = ((0, 100), (0,100))

# 2D Obstacle Ranges
obstacles = [((30, 40), (30, 40)), ((4, 7), (1, 3)), ((70, 90), (50, 60))]
n_obstacle = len(obstacles)

# Essentially reject points not in C_free
def is_collision(sample):
    sample_x, sample_y = sample
    for obstacle in obstacles:
        (obstacle_x_min, obstacle_x_max), (obstacle_y_min, obstacle_y_max) = obstacle
        if (sample_x >= obstacle_x_min and sample_x <= obstacle_x_max) and (sample_y >= obstacle_y_min and sample_y <= obstacle_y_max):
            return True
    return False

# Approximate linear interpolation 
# checker, you could also use 
# Liang-Barsky
def is_edge_collision(a, b):
    n = 1000 # rough approx

    a_np = np.array([a[0], a[1]])
    b_np = np.array([b[0], b[1]])

    for t in np.linspace(0, 1, n+1):
        y = a_np + t * (b_np - a_np)
        if (is_collision(y)): return True

    return False

def nearest_neighbors(p, P, k=3):
    # First k items -> knn
    return sorted(P.items(), key=lambda q: l2_dist(p, q[1]))[:k]

def PRM(q_start, q_goal, max_tries=1000):
    R = {"Start": [], "Goal": []}
    P = {"Start": q_start, "Goal": q_goal}

    for i in range(max_tries):
        sample_x = rng.uniform(low=C_space[0][0], high=C_space[0][1])
        sample_y = rng.uniform(low=C_space[1][0], high=C_space[1][1])

        sample = (sample_x, sample_y)
        if(is_collision(sample) or sample in P.values()): continue

        neighbors = nearest_neighbors(sample, P)

        n_samples = len(P) - 2
        q_name = "q" + str(n_samples)
        P[q_name] = (sample_x, sample_y)
        R[q_name] = []

        for neighbor in neighbors:
            if is_edge_collision(sample, neighbor[1]): continue
            d = l2_dist(sample, neighbor[1])
            R[q_name].append((neighbor[0], d))
            R[neighbor[0]].append((q_name, d))

        P[q_name] = sample

    return (R, P)


R, P = PRM((0, 0), (100, 100))

print(a_star(R, P, l2_dist))