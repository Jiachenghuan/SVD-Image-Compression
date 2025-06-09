from utils.image_loader import load_image_rgb
from utils.plot_and_export import plot_rgb_error_vs_compression, final_compress_with_chosen_k

if __name__ == "__main__":
    image_path = "demo.jpg"

    image_rgb = load_image_rgb(image_path)
    result = plot_rgb_error_vs_compression(image_rgb)

    if result[0] is not None:
        final_compress_with_chosen_k(image_rgb, *result)
