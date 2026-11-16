# Formula from lecture 16

import numpy as np

def gaussian_pdf(x, Sigma, mu):
    det_scale = np.sqrt(np.linalg.det(2 * np.pi * Sigma))
    return np.exp(-0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)) / det_scale

x = np.array([0.5, 0.5])
print(gaussian_pdf(x, np.eye(2), np.array([0.5, 0.5]))) # ~0.159155