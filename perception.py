# image and algorithms from lecture 15

import numpy as np
from PIL import Image

def to_grayscale(image):
    gray = np.dot(image[..., :3], [0.2989, 0.1140, 0.5870])
    return gray.astype(np.uint8)

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def rgb_mask(image, r_lim, g_lim, b_lim):
    image = image.copy()
    red_channel = image[:, :, 0]
    blue_channel = image[:, :, 2]
    green_channel = image[:, :, 1]

    red_mask = (red_channel >= r_lim[0]) & (red_channel <= r_lim[1])
    blue_mask = (blue_channel >= b_lim[0]) & (blue_channel <= b_lim[1])
    green_mask = (green_channel >= g_lim[0]) & (green_channel <= g_lim[1])

    mask = red_mask & blue_mask & green_mask

    image[~mask] = 0
    return image

img = np.asarray(Image.open("image.png"))
img_grayscale = to_grayscale(img)
red_mask = rgb_mask(img, (157, 255), (0, 99), (10, 193))
blue_mask = rgb_mask(img, (0, 64), (0, 143), (91, 255))

save_image(red_mask, "red_mask.png")
save_image(blue_mask, "blue_mask.png")

gaussian_filter = 0.1 * np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# discrete convolution
# note: we don't flip the kernel, so 
# this is technically cross-corr.
def conv(f, g, clip=True):
    z = np.zeros(f.shape)

    h, w, c = f.shape

    # To handle the border cases, zero-pad.
    f_pad = np.pad(f, ((1, 1), (1, 1), (0, 0)), mode="constant") 

    for i in range(c):
        for j in range(h):
            for k in range(w):
                patch = f_pad[j:j+3, k:k+3, i]
                z[j, k, i] = np.sum(patch * g)

    if clip: return np.clip(z, 0, 255).astype(np.uint8)
    return z

# discrete gray scale convolution;
# sobel filters are best with this version.
def conv_gray(f, g, clip=True):
    z = np.zeros(f.shape)

    h, w = f.shape

    # To handle the border cases, zero-pad.
    f_pad = np.pad(f, 1, mode="constant") 

    for j in range(h):
        for k in range(w):
            patch = f_pad[j:j+3, k:k+3]
            z[j, k] = np.sum(patch * g)

    if clip: return np.clip(z, 0, 255).astype(np.uint8)
    return z

def G(image):
    G_unscaled = np.sqrt(conv_gray(image, s_x, False)**2 + conv_gray(img_grayscale, s_y, False)**2)
    return np.clip(G_unscaled, 0, 255).astype(np.uint8)

def dialation_gray(image):
    z = np.zeros(image.shape)

    h, w = image.shape

    # To handle the border cases, zero-pad.
    image_pad = np.pad(image, 1, mode="constant") 

    for j in range(h):
        for k in range(w):
            z[j, k] = np.max(image_pad[j:j+3, k:k+3])

    return np.clip(z, 0, 255).astype(np.uint8)


def erosion_gray(image):
    z = np.zeros(image.shape)

    h, w = image.shape

    # To handle the border cases, zero-pad.
    image_pad = np.pad(image, 1, constant_values=255, mode="constant") 

    for j in range(h):
        for k in range(w):
            z[j, k] = np.min(image_pad[j:j+3, k:k+3])

    return np.clip(z, 0, 255).astype(np.uint8)

save_image(conv(img, gaussian_filter), "gaussian_blur.png")
save_image(conv_gray(img_grayscale, s_x), "dx.png")
save_image(conv_gray(img_grayscale, s_y), "dy.png")
save_image(G(img_grayscale), "G.png")
save_image(dialation_gray(img_grayscale), "dialation.png")
save_image(erosion_gray(img_grayscale), "erosion.png")