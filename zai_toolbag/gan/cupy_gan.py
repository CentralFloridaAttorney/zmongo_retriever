#!/usr/bin/env python3
import os
import re
import cupy as cp
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# ----------------------------
# Helper functions to create sample images
# ----------------------------
def get_next_model_filename(base_model_path, model_dir):
    """
    Given a base model file name (e.g., "gan_model_red_circle_v000.npz") and a directory,
    returns a new filename with an incremented version number that does not yet exist in model_dir.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(base_model_path)
    match = re.match(r"(.+)_v(\d+)(\.\w+)$", filename)
    if match:
        prefix, version_str, ext = match.groups()
        version = int(version_str)
        while True:
            version += 1
            new_filename = f"{prefix}_v{version:03d}{ext}"
            full_path = os.path.join(model_dir, new_filename)
            if not os.path.exists(full_path):
                return full_path
    else:
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_v000{ext}"
        full_path = os.path.join(model_dir, new_filename)
        if not os.path.exists(full_path):
            return full_path
        else:
            version = 0
            while True:
                version += 1
                new_filename = f"{base}_v{version:03d}{ext}"
                full_path = os.path.join(model_dir, new_filename)
                if not os.path.exists(full_path):
                    return full_path

# ----------------------------
# Standard CupyGAN class
# ----------------------------
class CupyGAN:
    def __init__(self, image_path, noise_dim=100, lr_D=0.001, lr_G=0.001, target_size=None):
        """
        Initializes the GAN:
          - Loads a full-color image from image_path.
          - Optionally resizes the image to target_size (width, height).
          - Scales pixel values to [-1,1] and flattens the image.
          - Initializes generator and discriminator parameters.

        Parameters:
          image_path (str): Path to the training image.
          noise_dim (int): Dimension of the noise vector.
          lr_D (float): Learning rate for the discriminator.
          lr_G (float): Learning rate for the generator.
          target_size (tuple): Optional (width, height) to resize the image.
        """
        img = Image.open(image_path).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size)
        self.img = img
        self.width, self.height = img.size
        self.channels = 3

        self.image_size = self.width * self.height * self.channels
        print(f"Loaded image: {image_path}")
        print(f"Dimensions: {self.width} x {self.height}, Channels: {self.channels}, Total size: {self.image_size}")

        img_np = np.array(img).astype(np.float32)
        img_np = (img_np / 127.5) - 1.0
        self.x_real = cp.array(img_np.flatten())

        self.noise_dim = noise_dim
        self.lr_D = lr_D
        self.lr_G = lr_G

        self.W_G = cp.random.randn(self.image_size, self.noise_dim).astype(cp.float32) * 0.02
        self.b_G = cp.zeros((self.image_size,), dtype=cp.float32)

        self.w_D = cp.random.randn(self.image_size).astype(cp.float32) * 0.02
        self.b_D = cp.array(0.0, dtype=cp.float32)

    def sigmoid(self, x):
        return 1.0 / (1.0 + cp.exp(-x))

    def generator_forward(self, z):
        v = cp.dot(self.W_G, z) + self.b_G
        return cp.tanh(v)

    def discriminator_forward(self, x):
        u = cp.dot(self.w_D, x) + self.b_D
        return self.sigmoid(u)

    def train_step(self):
        z = cp.random.randn(self.noise_dim).astype(cp.float32)
        x_fake = self.generator_forward(z)
        D_real = self.discriminator_forward(self.x_real)
        D_fake = self.discriminator_forward(x_fake)
        loss_D = -cp.log(D_real + 1e-8) - cp.log(1 - D_fake + 1e-8)
        loss_G = -cp.log(D_fake + 1e-8)

        grad_u_real = -(1 - D_real)
        grad_u_fake = D_fake
        grad_w_D = grad_u_real * self.x_real + grad_u_fake * x_fake
        grad_b_D = grad_u_real + grad_u_fake
        self.w_D -= self.lr_D * grad_w_D
        self.b_D -= self.lr_D * grad_b_D

        grad_u_fake_for_G = -(1 - D_fake)
        grad_x_fake = grad_u_fake_for_G * self.w_D
        d_tanh = 1 - x_fake * x_fake
        grad_v = grad_x_fake * d_tanh
        grad_W_G = cp.outer(grad_v, z)
        grad_b_G = grad_v
        self.W_G -= self.lr_G * grad_W_G
        self.b_G -= self.lr_G * grad_b_G

        return float(loss_D), float(loss_G)

    def generate_image(self):
        z = cp.random.randn(self.noise_dim).astype(cp.float32)
        x_fake = self.generator_forward(z)
        x_fake_np = cp.asnumpy(x_fake)
        img_generated = x_fake_np.reshape((self.height, self.width, self.channels))
        img_generated = ((img_generated + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_generated)

    def save_model(self, file_path):
        np.savez(file_path,
                 W_G=cp.asnumpy(self.W_G),
                 b_G=cp.asnumpy(self.b_G),
                 w_D=cp.asnumpy(self.w_D),
                 b_D=float(self.b_D))
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            data = np.load(file_path)
            if data["W_G"].shape[0] != self.image_size:
                print("Saved model dimensions do not match current image. Skipping load.")
                return
            self.W_G = cp.array(data["W_G"], dtype=cp.float32)
            self.b_G = cp.array(data["b_G"], dtype=cp.float32)
            self.w_D = cp.array(data["w_D"], dtype=cp.float32)
            self.b_D = cp.array(data["b_D"], dtype=cp.float32)
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model file {file_path} not found.")

