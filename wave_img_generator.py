import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import matplotlib.cm as cm
import datetime
import hashlib
import random
import pathlib


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
    for y in range(height):
        t = y / height
        color = tuple(int(start_color[i] * (1 - t) + end_color[i] * t) for i in range(3))
        canvas[y, :, :] = color
    return canvas


def draw_sine_wave(canvas, x_offset, y_pos, amplitude, frequency, color, thickness):
    """Draw a single sine wave."""
    height, width, _ = canvas.shape
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    points = []
    for x in range(width):
        y = int(y_pos * height - amplitude * height * math.sin(2 * math.pi * frequency * (x / width - x_offset)))
        points.append((x, y))

    draw.line(points, fill=color, width=thickness)
    return np.array(img)


def apply_blur_by_depth(canvas, focal_y, max_blur_radius):
    """Apply blur based on distance from focal plane."""
    height, width, _ = canvas.shape
    img = Image.fromarray(canvas)

    # Create blur map - no blur before focal point, quick falloff after
    y_coords = np.arange(height)
    distance_after_focal = np.maximum(0, -y_coords + focal_y * height) / height
    blur_amounts = np.clip(distance_after_focal * max_blur_radius * 8, 0, max_blur_radius)

    # Pre-compute blurred versions at integer blur levels
    blurred_images = {}
    for level in range(0, max_blur_radius + 1):
        blurred_images[level] = np.array(img.filter(ImageFilter.GaussianBlur(radius=level)))

    # Apply progressive blur with interpolation
    result = np.array(img)
    for y in range(height):
        target_blur = blur_amounts[y]
        lower = int(np.floor(target_blur))
        upper = int(np.ceil(target_blur))

        if lower == upper:
            result[y] = blurred_images[lower][y]
        else:
            # Blend between two blur levels
            t = target_blur - lower
            result[y] = ((1 - t) * blurred_images[lower][y] + t * blurred_images[upper][y]).astype(np.uint8)

    print(f"📸 Applied depth blur (focal plane at {focal_y:.1%})")
    return result


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
    # Create canvas with gradient background
    canvas = create_canvas(dpi=200)
    canvas = fill_gradient(canvas, start_color=(0, 0, 00), end_color=(128, 231, 254))

    # Draw waves with color gradient
    cmap = cm.get_cmap("managua")
    for i in np.arange(0, 1, 0.0025):
        color = tuple(int(255 * c) for c in cmap(i)[:3])
        canvas = draw_sine_wave(
            canvas,
            x_offset=i,
            y_pos=i * 0.75 + 0.1,
            amplitude=0.15,
            frequency=3.5,
            color=color,
            thickness=40
        )

    # Apply bokeh effect (focal point at 60% from top)
    canvas = apply_blur_by_depth(canvas, focal_y=0.6, max_blur_radius=30)

    save_image(canvas, prefix="sine_wave_bokeh")