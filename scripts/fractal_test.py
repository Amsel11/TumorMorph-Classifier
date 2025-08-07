import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(current_dir, 'scripts')
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
from MorphFeatureClass import TumorMorphology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fa = FractalAnalyzer3D(device=device)

def generate_smooth_sphere(shape=(128, 128, 128), radius=30):
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) // 2
    sphere = ((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2) < radius**2
    return sphere.astype(np.uint8)

def generate_noisy_blob(shape=(128, 128, 128)):
    blob = np.random.rand(*shape) > 0.995
    return gaussian_filter(blob.astype(float), sigma=2) > 0.1

def generate_fractal_noise(shape=(128, 128, 128)):
    # Sum of multiple Gaussian blobs = pseudo fractal
    np.random.seed(42)
    noise = np.zeros(shape)
    for _ in range(100):
        x, y, z = np.random.randint(0, shape[0], 3)
        blob = np.zeros(shape)
        blob[x, y, z] = 1
        noise += gaussian_filter(blob, sigma=5)
    return noise > np.percentile(noise, 99)

def compute_and_print_fd(mask_np, name):
    mask_torch = torch.tensor(mask_np).to(device).unsqueeze(0).unsqueeze(0).float()
    fd, r2, scale_range, points = fa(mask_torch)
    print(f"{name} | FD = {fd:.4f}, R² = {r2:.4f}, Scale range = {scale_range}, Points = {points}")

# Generate and evaluate
sphere = generate_smooth_sphere()
noisy_blob = generate_noisy_blob()
fractal_blob = generate_fractal_noise()

compute_and_print_fd(sphere, "Smooth Sphere")
compute_and_print_fd(noisy_blob, "Noisy Blob")
compute_and_print_fd(fractal_blob, "Fractal-Like Noise")

# Optional: visualize central slices
def show_slice(mask, name):
    plt.imshow(mask[mask.shape[0] // 2], cmap='gray')
    plt.title(name)
    plt.axis('off')
    plt.show()

show_slice(sphere, "Smooth Sphere Slice")
show_slice(noisy_blob, "Noisy Blob Slice")
show_slice(fractal_blob, "Fractal-Like Noise Slice")
