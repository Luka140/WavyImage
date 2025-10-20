import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime
import hashlib
import random
import pathlib
from PIL import ImageFilter



def create_a4_grid_array(rows=20, cols=10, dpi=300):
    """Create a blank A4-sized grid as a NumPy array (white background)."""
    a4_width_mm = 210
    a4_height_mm = 297
    mm_to_inch = 1 / 25.4
    width_px = int(a4_width_mm * mm_to_inch * dpi)
    height_px = int(a4_height_mm * mm_to_inch * dpi)

    grid = np.ones((height_px, width_px, 3), dtype=np.uint8) * 255
    cell_h = height_px // rows
    cell_w = width_px // cols

    print(f"✅ Created grid array: {rows}×{cols} cells, each {cell_w}×{cell_h}px")
    return grid, cell_h, cell_w


def set_cell_color(grid, row, col, cell_h, cell_w, color):
    """Fill a specific cell (by row, col) with a color."""
    y1, y2 = row * cell_h, (row + 1) * cell_h
    x1, x2 = col * cell_w, (col + 1) * cell_w
    grid[y1:y2, x1:x2] = color


def draw_sine_wave(grid, x_offset=0, y_pos=0.5, amplitude=0.4, frequency=2, color=(0, 0, 255), thickness=3):
    """
    Draw a sine wave across the grid.
    - amplitude: fraction of half the image height (0.4 = 40%)
    - frequency: number of sine cycles across the page width
    """
    height, width, _ = grid.shape
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)

    y_position = int(height * y_pos)
    amp_px = int(amplitude * height / 2)

    # Compute points
    points = []
    for x in range(width):
        norm_x = x / width
        y = y_position - int(math.sin(2 * math.pi * frequency * (norm_x - x_offset)) * amp_px)
        points.append((x, y))

    draw.line(points, fill=color, width=thickness)

    return np.array(img)

def fill_gradient(grid, cmap_name="viridis", direction="vertical"):
    """
    Fill the entire grid with a smooth gradient using a matplotlib colormap.
    
    Args:
        grid (np.ndarray): The A4-sized numpy array (RGB image).
        cmap_name (str): Name of the matplotlib colormap to use.
        direction (str): "vertical" or "horizontal" gradient.
    Returns:
        np.ndarray: Updated grid with gradient applied.
    """
    height, width, _ = grid.shape
    cmap = cm.get_cmap(cmap_name)

    # Generate normalized gradient values (0 → 1)
    if direction == "vertical":
        gradient = np.linspace(0, 1, height)
        colors = (cmap(gradient)[:, :3] * 255).astype(np.uint8)
        for y in range(height):
            grid[y, :, :] = colors[y]
    elif direction == "horizontal":
        gradient = np.linspace(0, 1, width)
        colors = (cmap(gradient)[:, :3] * 255).astype(np.uint8)
        for x in range(width):
            grid[:, x, :] = colors[x]
    else:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    print(f"🎨 Filled grid with {direction} gradient using colormap '{cmap_name}'.")
    return grid

def apply_lens_blur(
    grid,
    focus_row,
    focus_col,
    cell_h,
    cell_w,
    max_radius=15,
    falloff=0.002,
    levels=5,
    elliptical=False,
    aspect_ratio=1.0
):
    """
    Simulate a lens-like blur (depth-of-field) centered at a given grid cell.

    Args:
        grid (np.ndarray): The RGB image array.
        focus_row (int): Row index of the focal point (sharpest area).
        focus_col (int): Column index of the focal point.
        cell_h (int): Height of one grid cell.
        cell_w (int): Width of one grid cell.
        max_radius (int): Maximum blur radius at the image edge.
        falloff (float): Controls how quickly blur increases with distance.
        levels (int): Number of pre-blurred layers to blend for realism.
        elliptical (bool): If True, makes blur falloff elliptical.
        aspect_ratio (float): Ellipticity ratio if elliptical=True (e.g., 1.5 for wide bokeh).

    Returns:
        np.ndarray: Image with realistic lens blur effect.
    """
    height, width, _ = grid.shape
    focus_x = int(focus_col * cell_w + cell_w / 2)
    focus_y = int(focus_row * cell_h + cell_h / 2)

    base = Image.fromarray(grid).convert("RGB")

    # Generate distance map
    y, x = np.ogrid[:height, :width]
    dx = (x - focus_x).astype(np.float32)
    dy = (y - focus_y).astype(np.float32)

    if elliptical:
        dy *= float(aspect_ratio)  # stretch vertically or horizontally

    distance = np.sqrt(dx ** 2 + dy ** 2)


    # Normalize distance and compute blur amount
    distance_norm = np.clip(distance * falloff, 0, 1)
    blur_strength = distance_norm ** 2  # quadratic for smooth falloff

    # Precompute multiple blurred versions
    blurred_layers = [
        base.filter(ImageFilter.GaussianBlur(radius=max_radius * (i / (levels - 1))))
        for i in range(levels)
    ]

    # Blend according to distance
    blended = np.zeros((height, width, 3), dtype=np.float32)
    blur_indices = (blur_strength * (levels - 1)).astype(int)

    for i in range(levels):
        mask = (blur_indices == i)
        blurred_np = np.array(blurred_layers[i], dtype=np.float32)
        blended[mask] = blurred_np[mask]

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    print(f"📸 Applied lens-style blur centered at ({focus_row}, {focus_col})")
    return blended

def save_grid_as_png(grid, filename_prefix="a4_sine_wave", directory="figures", use_hash=False):
    """Save the grid array as a PNG file with a unique timestamp or hash."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if use_hash:
        # Generate short hash from random seed + timestamp
        random_bits = f"{random.random()}_{timestamp}".encode("utf-8")
        unique_id = hashlib.sha1(random_bits).hexdigest()[:6]
        filename = f"{filename_prefix}_{unique_id}.png"
    else:
        filename = f"{filename_prefix}_{timestamp}.png"

    path = pathlib.Path(__file__).parent / directory / filename
    img = Image.fromarray(grid)
    img.save(path)
    print(f"💾 Saved at {path}")


def random_wave_parameters():
    """
    Generate a set of randomized but visually reasonable parameters
    for the sine wave pattern.
    """
    step = round(random.uniform(0.02, 0.1), 3)        # how densely to draw waves
    frequency = random.uniform(1.0, 6.0)              # number of cycles across the page
    amplitude = random.uniform(0.1, 0.5)              # fraction of half-page height
    thickness = random.randint(5, 20)                 # line thickness in pixels
    x_offset = random.uniform(-0.5, 0.5)              # phase shift
    y_pos = random.uniform(0.1, 0.9)                  # vertical placement (0–1)
    
    print(f"🎲 Randomized parameters:")
    print(f"  step={step}, freq={frequency:.2f}, amp={amplitude:.2f}, "
          f"thickness={thickness}, x_offset={x_offset:.2f}, y_pos={y_pos:.2f}")
    
    return step, frequency, amplitude, thickness, x_offset, y_pos



if __name__ == "__main__":
    grid, cell_h, cell_w = create_a4_grid_array(rows=20, cols=10, dpi=200)

    grid = fill_gradient(grid, 'gray')

    cmap = cm.get_cmap("magma")   # try 'turbo' 'viridis', 'plasma', 'magma', 'inferno', etc.

    start, stop = 0, 1 
    steps = 0.0025
    rows = stop // steps
    for i in np.arange(0, 1, steps):
        rgba = cmap(i)[:3]
        color = tuple(int(255 * c) for c in rgba)
        grid = draw_sine_wave(
            grid,
            x_offset=i,
            y_pos=i * 0.75 + 0.1,
            amplitude=0.15,
            frequency=3.5,
            color=color,
            thickness=40
        )

    grid = apply_lens_blur(
        grid,
        focus_row=int(rows*0.8),
        focus_col=5,
        cell_h=cell_h,
        cell_w=cell_w,
        max_radius=10,
        falloff=0.0015,
        levels=7,
        elliptical=True,
        aspect_ratio=1.3
    )
    # Automatically unique file name
    save_grid_as_png(grid, filename_prefix="a4_sine_wave", directory='figures', use_hash=True)


    # Generate random parameters

    # for i in range(100):
    #     grid, cell_h, cell_w = create_a4_grid_array(rows=20, cols=10)
    #     step, frequency, amplitude, thickness, x_offset, y_pos = random_wave_parameters()

    #     for i in np.arange(0, 1, step):
    #         rgba = cmap(i)[:3]
    #         color = tuple(int(255 * c) for c in rgba)
    #         grid = draw_sine_wave(
    #             grid,
    #             x_offset=x_offset + i,   # add a phase offset per wave
    #             y_pos=y_pos + i * 0.3,   # small vertical drift
    #             amplitude=amplitude,
    #             frequency=frequency,
    #             color=color,
    #             thickness=thickness
    #         )


    #     # Automatically unique file name
    #     save_grid_as_png(grid, filename_prefix="a4_sine_wave", directory='figures/random_figures', use_hash=True)

