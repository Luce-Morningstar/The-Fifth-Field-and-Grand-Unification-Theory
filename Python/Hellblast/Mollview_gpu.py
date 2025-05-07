import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def mollview_gpu(sphere_map, nside=64, title="", cbar=True, output_file=None):
    npix = 12 * nside**2
    if sphere_map.shape[0] != npix:
        raise ValueError(f"Expected {npix} pixels for nside {nside}, got {sphere_map.shape[0]}")

    log_data = cp.log1p(cp.asarray(sphere_map)).get()  # move to CPU for plotting

    plt.figure(figsize=(10, 5))
    im = plt.imshow(log_data.reshape((12 * nside, int(npix / (12 * nside)))), cmap='inferno', origin='lower')
    plt.title(title)
    if cbar:
        plt.colorbar(im, orientation='horizontal')
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()
