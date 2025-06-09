import numpy as np

def svd_compress_channel(channel, k):
    U, S, VT = np.linalg.svd(channel, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

def svd_compress_rgb(image_rgb, k):
    return np.stack([
        svd_compress_channel(image_rgb[:, :, i], k)
        for i in range(3)
    ], axis=2)

def compute_rgb_compression_ratio(m, n, k):
    original_size = m * n * 3 * 4
    compressed_size = 3 * (m * k * 4 + k * 4 + k * n * 4)
    return compressed_size / original_size
