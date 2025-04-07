import os

from PIL import Image, ImageDraw


def create_red_circle_image(path, size=(512, 512)):
    """Creates a red circle on a white background."""
    if not os.path.exists(path):
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        center = (size[0] // 2, size[1] // 2)
        radius = 20
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(bbox, fill="red", outline="red")
        img.save(path)
        print(f"Saved red circle image to {path}")
    else:
        print(f"Red circle image already exists: {path}")


def create_blue_square_image(path, size=(512, 512)):
    """Creates a blue square on a white background."""
    if not os.path.exists(path):
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        margin = 20
        draw.rectangle([margin, margin, size[0] - margin, size[1] - margin],
                       fill="blue", outline="blue")
        img.save(path)
        print(f"Saved blue square image to {path}")
    else:
        print(f"Blue square image already exists: {path}")


def create_blue_circle_image(path, size=(512, 512)):
    """Creates a blue circle on a white background."""
    if not os.path.exists(path):
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        center = (size[0] // 2, size[1] // 2)
        radius = 20
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(bbox, fill="blue", outline="blue")
        img.save(path)
        print(f"Saved blue circle image to {path}")
    else:
        print(f"Blue circle image already exists: {path}")


def create_sample_image(sample, path, size=(512, 512)):
    """Creates a sample image based on the sample type."""
    if sample == "red_circle":
        create_red_circle_image(path, size)
    elif sample == "blue_square":
        create_blue_square_image(path, size)
    elif sample == "blue_circle":
        create_blue_circle_image(path, size)
    else:
        print(f"Unknown sample type '{sample}'. Creating red circle as default.")
        create_red_circle_image(path, size)


