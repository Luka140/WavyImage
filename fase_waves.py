import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import matplotlib.cm as cm
import datetime
import hashlib
import random
import pathlib
from scipy.ndimage import gaussian_filter


def create_canvas(dpi=200):
    """Create a blank A4-sized canvas (white background)."""
    width_px = int(210 / 25.4 * dpi)
    height_px = int(297 / 25.4 * dpi)
    canvas = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    print(f"✅ Created {width_px}×{height_px}px canvas")
    return canvas


def fill_gradient(canvas, start_color=(0, 0, 0), end_color=(255, 255, 255)):
    """Fill canvas with vertical gradient."""
    height, width, _ = canvas.shape
    # Vectorized gradient calculation
    t = np.linspace(0, 1, height).reshape(-1, 1, 1)
    gradient = np.array(start_color) * (1 - t) + np.array(end_color) * t
    canvas[:] = gradient.astype(np.uint8)
    return canvas


def draw_all_waves(canvas, cmap, step=0.0025):
    """Draw all sine waves at once on a single PIL image."""
    height, width, _ = canvas.shape
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    # Draw all waves in one PIL session
    for i in np.arange(0, 1, step):
        color = tuple(int(255 * c) for c in cmap(i)[:3])
        y_pos = i * 0.75 + 0.1

        # Precompute all points for this wave
        x = np.arange(width)
        y = (y_pos * height - 0.15 * height * np.sin(2 * math.pi * 3.5 * (x / width - i))).astype(int)
        points = list(zip(x.tolist(), y.tolist()))

        draw.line(points, fill=color, width=40)

    return np.array(img)


def apply_blur_by_depth(canvas, focal_y, max_blur_radius):
    """Apply blur based on distance from focal plane using scipy."""
    height, width, _ = canvas.shape

    # Compute blur amount for each row
    y_coords = np.arange(height)
    blur_amounts = np.abs(y_coords - focal_y * height) / height
    blur_amounts = blur_amounts * max_blur_radius * 2

    # Apply variable blur per channel using scipy
    result = canvas.copy().astype(np.float32)
    for channel in range(3):
        for y in range(height):
            sigma = blur_amounts[y]
            if sigma > 0.5:
                # Blur a small region around this row
                y_start = max(0, y - int(sigma * 3))
                y_end = min(height, y + int(sigma * 3) + 1)
                region = result[y_start:y_end, :, channel]
                blurred = gaussian_filter(region, sigma=(sigma, sigma))
                result[y, :, channel] = blurred[y - y_start, :]

    print(f"📸 Applied depth blur (focal plane at {focal_y:.1%})")
    return np.clip(result, 0, 255).astype(np.uint8)


def save_image(canvas, prefix="artwork", directory="figures"):
    """Save canvas with unique filename."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = hashlib.sha1(f"{random.random()}_{timestamp}".encode()).hexdigest()[:6]
    filename = f"{prefix}_{unique_id}.png"

    path = pathlib.Path(__file__).parent / directory / filename
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(canvas).save(path)
    print(f"💾 Saved: {path}")


if __name__ == "__main__":
    canvas = create_canvas(dpi=600)
    canvas = fill_gradient(canvas, start_color=(20, 20, 20), end_color=(255, 255, 195))

    cmap = cm.get_cmap("magma")
    canvas = draw_all_waves(canvas, cmap, step=0.0025)

    canvas = apply_blur_by_depth(canvas, focal_y=0.85, max_blur_radius=36)

    save_image(canvas, prefix="sine_wave_bokeh")