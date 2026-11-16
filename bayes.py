# Example & algorithm from lecture 18

import numpy as np

# This implements a discrete-time 
# Bayes filter. You'll need to discretize 
# your states to use this for continuous 
# cases.
def bayes_filter(bel_prev, u, z, P, Z):
    # Marginalize other all previous states
    # for prediction step using preiction model
    bel_pred = (P[u] @ bel_prev).reshape(-1)
    bel_unnormalized = Z[z] * bel_pred
    eta = np.sum(bel_unnormalized)

    return bel_unnormalized / eta # exactly bel


# This is the door robot example from lecture 18
bel = np.array([0.5, 0.5])

# The prediction / state transition model, cardinality |U| x |S| x |S|
P = np.array([
        [
            [[1.0, 0.8], 
             [0.0, 0.2]]
        ], 
        [
            [[1.0, 0.0],
             [0.0, 1.0]]
        ]
    ])

# The measurement model, cardinality |Z| x |S|
Z = np.array(
        [[0.6, 0.2],
        [0.4, 0.8]]
    )

T = 2
action_sequence = [1, 0]
measurement_sequence = [0, 0]

for t in range(T):
    u = action_sequence[t]
    z = measurement_sequence[t]

    bel = bayes_filter(bel, u, z, P, Z)

print(bel) # Should be ~ (0.983, 0.017)