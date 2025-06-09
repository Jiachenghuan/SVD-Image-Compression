import matplotlib.pyplot as plt
import pandas as pd
from utils.svd_core import svd_compress_rgb, compute_rgb_compression_ratio
from utils.metrics import compute_rgb_relative_error, compute_rgb_psnr, compute_rgb_ssim, compute_rgb_lpips


def find_best_k_auto(ratios, errors, psnrs, ssims, lpips_vals, ks,
                     min_psnr=30, min_ssim=0.92, max_lpips=0.08, max_error=0.1):
    for i in range(len(ks)):
        if psnrs[i] >= min_psnr and ssims[i] >= min_ssim and lpips_vals[i] <= max_lpips and errors[i] <= max_error and \
                ratios[i] < 1:
            return ks[i], ratios[i], errors[i], psnrs[i], ssims[i], lpips_vals[i]
    return None, None, None, None, None, None


def plot_rgb_error_vs_compression(image_rgb, output_csv_path="results/compression_metrics.csv"):
    m, n, _ = image_rgb.shape
    k_max = int((m * n) / (m + n + 1))
    step = max(1, k_max // 50)
    ks = list(range(5, k_max, step))

    ratios, errors, psnrs, ssims, lpips_values = [], [], [], [], []

    for k in ks:
        cmp = svd_compress_rgb(image_rgb, k)
        ratios.append(compute_rgb_compression_ratio(m, n, k))
        errors.append(compute_rgb_relative_error(image_rgb, cmp))
        psnrs.append(compute_rgb_psnr(image_rgb, cmp))
        ssims.append(compute_rgb_ssim(image_rgb, cmp))
        lpips_values.append(compute_rgb_lpips(image_rgb, cmp))

    pd.DataFrame({
        "k": ks,
        "Compression Ratio": ratios,
        "Frobenius Relative Error": errors,
        "PSNR (dB)": psnrs,
        "SSIM": ssims,
        "LPIPS": lpips_values
    }).to_csv(output_csv_path, index=False)

    plt.figure(figsize=(18, 10))
    titles = ["Frobenius Error", "PSNR", "SSIM", "LPIPS"]
    y_data = [errors, psnrs, ssims, lpips_values]
    colors = ['purple', 'blue', 'red', 'green']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(ratios, y_data[i], marker='o', color=colors[i])
        plt.xlabel("Compression Ratio")
        plt.ylabel(titles[i])
        plt.title(f"{titles[i]} vs Compression Ratio")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    return find_best_k_auto(ratios, errors, psnrs, ssims, lpips_values, ks)


def final_compress_with_chosen_k(image_rgb, k, ratio, error, psnr, ssim_val, lpips_val):
    print(f"\nUsing k = {k} for compression:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim_val:.3f}")
    print(f"  LPIPS = {lpips_val:.3f}")
    print(f"  Error = {error:.3f}")
    print(f"  Compression Ratio = {ratio:.3f}")

    from utils.svd_core import svd_compress_rgb
    compressed = svd_compress_rgb(image_rgb, k)
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb.astype(np.uint8))
    plt.title("Original")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(compressed, 0, 255).astype(np.uint8))
    plt.title(f"Compressed (k={k})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
