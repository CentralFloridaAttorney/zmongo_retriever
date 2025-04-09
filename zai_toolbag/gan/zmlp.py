#!/usr/bin/env python3
import os
import re
import argparse
import cupy as cp
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# ----------------------------
# Helper functions to create sample images
# ----------------------------
def create_red_circle_image(path, size=(512, 512)):
    """Creates a red circle on a white background."""
    if not os.path.exists(path):
        img = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(img)
        center = (size[0] // 2, size[1] // 2)
        radius = 40  # adjusted for larger image
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
        margin = 40
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
        radius = 40
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


# ----------------------------
# Helper function: Generate a new sequential model filename in a given directory
# ----------------------------
def get_next_model_filename(base_model_path, model_dir):
    """
    Given a base model file name (e.g., "gan_model_v000.npz") and a directory,
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
# MLP class for defining multi-layer perceptrons
# ----------------------------
class MLP:
    def __init__(self, sizes, hidden_activation="tanh", output_activation="tanh", weight_scale=0.02):
        """
        sizes: list of integers, e.g. [input_dim, hidden1, ..., output_dim]
        hidden_activation: activation for hidden layers.
        output_activation: activation for output layer.
        """
        self.sizes = sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_layers = len(sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            W = cp.random.randn(sizes[i], sizes[i + 1]).astype(cp.float32) * weight_scale
            b = cp.zeros((sizes[i + 1],), dtype=cp.float32)
            self.weights.append(W)
            self.biases.append(b)

    def activate(self, a, act_type):
        if act_type == "tanh":
            return cp.tanh(a)
        elif act_type == "sigmoid":
            return 1.0 / (1.0 + cp.exp(-a))
        else:
            return a

    def activation_derivative(self, activated, act_type):
        if act_type == "tanh":
            return 1 - activated * activated
        elif act_type == "sigmoid":
            return activated * (1 - activated)
        else:
            return cp.ones_like(activated)

    def forward(self, x):
        activations = [x]
        pre_activations = []
        for i in range(self.num_layers):
            a = cp.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(a)
            if i < self.num_layers - 1:
                z = self.activate(a, self.hidden_activation)
            else:
                z = self.activate(a, self.output_activation)
            activations.append(z)
        return activations, pre_activations

    def backward(self, activations, pre_activations, grad_output):
        """Performs backpropagation; grad_output is dLoss/d(activation of output layer)."""
        grad_weights = [None] * self.num_layers
        grad_biases = [None] * self.num_layers
        grad = grad_output  # initial gradient from loss w.r.t. output activation
        for i in reversed(range(self.num_layers)):
            # Determine activation type for layer i
            act_type = self.output_activation if i == self.num_layers - 1 else self.hidden_activation
            deriv = self.activation_derivative(activations[i + 1], act_type)
            delta = grad * deriv
            grad_weights[i] = cp.outer(activations[i], delta)
            grad_biases[i] = delta
            grad = cp.dot(delta, self.weights[i].T)
        return grad_weights, grad_biases

    def backward_input(self, activations, pre_activations, grad_output):
        """Computes gradient with respect to the input of the MLP."""
        grad = grad_output
        for i in reversed(range(self.num_layers)):
            act_type = self.output_activation if i == self.num_layers - 1 else self.hidden_activation
            deriv = self.activation_derivative(activations[i + 1], act_type)
            delta = grad * deriv
            grad = cp.dot(delta, self.weights[i].T)
        return grad

    def update_params(self, grad_weights, grad_biases, lr):
        for i in range(self.num_layers):
            self.weights[i] -= lr * grad_weights[i]
            self.biases[i] -= lr * grad_biases[i]


# ----------------------------
# GAN class using MLPs for generator and discriminator
# ----------------------------
class CupyGAN:
    def __init__(self, image_path, noise_dim=100, lr_D=0.001, lr_G=0.001, target_size=None,
                 gen_layers=None, disc_layers=None):
        """
        Initializes the GAN:
          - Loads a full-color image from image_path.
          - Optionally resizes the image to target_size.
          - Scales pixel values to [-1,1] and flattens the image.
          - Builds MLPs for the generator and discriminator.

        Parameters:
          image_path (str): Path to the training image.
          noise_dim (int): Dimension of the noise vector.
          lr_D (float): Learning rate for the discriminator.
          lr_G (float): Learning rate for the generator.
          target_size (tuple): Optional (width, height) to resize the image.
          gen_layers (list): Optional list of layer sizes for the generator.
          disc_layers (list): Optional list of layer sizes for the discriminator.
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

        # Default architectures if not provided.
        if gen_layers is None:
            gen_layers = [noise_dim, 256, self.image_size]
        if disc_layers is None:
            disc_layers = [self.image_size, 256, 1]

        self.gen = MLP(gen_layers, hidden_activation="tanh", output_activation="tanh")
        self.disc = MLP(disc_layers, hidden_activation="tanh", output_activation="sigmoid")

    def train_step(self):
        # Generator forward pass
        z = cp.random.randn(self.noise_dim).astype(cp.float32)
        gen_acts, gen_pre = self.gen.forward(z)
        x_fake = gen_acts[-1]

        # Discriminator forward pass on real image
        real_acts, real_pre = self.disc.forward(self.x_real)
        D_real = real_acts[-1]  # scalar
        # Discriminator forward pass on fake image
        fake_acts, fake_pre = self.disc.forward(x_fake)
        D_fake = fake_acts[-1]  # scalar

        loss_D = -cp.log(D_real + 1e-8) - cp.log(1 - D_fake + 1e-8)
        loss_G = -cp.log(D_fake + 1e-8)

        # Backprop for discriminator:
        grad_output_real = -1 / (D_real + 1e-8)
        grad_output_fake = 1 / (1 - D_fake + 1e-8)
        gradW_real, gradb_real = self.disc.backward(real_acts, real_pre, grad_output_real)
        gradW_fake, gradb_fake = self.disc.backward(fake_acts, fake_pre, grad_output_fake)
        # Sum gradients (treat real and fake contributions)
        gradW_disc = [(gr + gf) for gr, gf in zip(gradW_real, gradW_fake)]
        gradb_disc = [(br + bf) for br, bf in zip(gradb_real, gradb_fake)]
        self.disc.update_params(gradW_disc, gradb_disc, self.lr_D)

        # Backprop for generator:
        # The loss for generator: -log(D_fake)
        grad_output_for_gen = -1 / (D_fake + 1e-8)
        # Compute gradient with respect to discriminator input (x_fake)
        grad_disc_input = self.disc.backward_input(fake_acts, fake_pre, grad_output_for_gen)
        # Backprop through generator
        gen_gradW, gen_gradb = self.gen.backward(gen_acts, gen_pre, grad_disc_input)
        self.gen.update_params(gen_gradW, gen_gradb, self.lr_G)

        return float(loss_D), float(loss_G)

    def generate_image(self):
        z = cp.random.randn(self.noise_dim).astype(cp.float32)
        gen_acts, _ = self.gen.forward(z)
        x_fake = gen_acts[-1]
        x_fake_np = cp.asnumpy(x_fake)
        img_generated = x_fake_np.reshape((self.height, self.width, self.channels))
        img_generated = ((img_generated + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_generated)

    def save_model(self, file_path):
        # Save both generator and discriminator parameters
        gen_params = {"weights": [cp.asnumpy(W) for W in self.gen.weights],
                      "biases": [cp.asnumpy(b) for b in self.gen.biases]}
        disc_params = {"weights": [cp.asnumpy(W) for W in self.disc.weights],
                       "biases": [cp.asnumpy(b) for b in self.disc.biases]}
        np.savez(file_path, gen_params=gen_params, disc_params=disc_params)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            gen_params = data["gen_params"].item()
            disc_params = data["disc_params"].item()
            # Check that dimensions match:
            if gen_params["weights"][0].shape[0] != self.noise_dim:
                print("Saved generator dimensions do not match current model. Skipping load.")
                return
            # Load generator parameters:
            self.gen.weights = [cp.array(W, dtype=cp.float32) for W in gen_params["weights"]]
            self.gen.biases = [cp.array(b, dtype=cp.float32) for b in gen_params["biases"]]
            # Load discriminator parameters:
            self.disc.weights = [cp.array(W, dtype=cp.float32) for W in disc_params["weights"]]
            self.disc.biases = [cp.array(b, dtype=cp.float32) for b in disc_params["biases"]]
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model file {file_path} not found.")


# ----------------------------
# CombinedCupyGAN: Combine multiple saved models by averaging their parameters
# ----------------------------
class CombinedCupyGAN(CupyGAN):
    def __init__(self, image_path, model_paths, noise_dim=100, lr_D=0.001, lr_G=0.001, target_size=None,
                 gen_layers=None, disc_layers=None):
        super().__init__(image_path, noise_dim, lr_D, lr_G, target_size=target_size,
                         gen_layers=gen_layers, disc_layers=disc_layers)
        combined_params = self.combine_models(model_paths)
        # Update generator parameters
        self.gen.weights = combined_params["W_G"]
        self.gen.biases = combined_params["b_G"]
        # Update discriminator parameters
        self.disc.weights = combined_params["w_D"]
        self.disc.biases = combined_params["b_D"]
        print("Combined models from:")
        for path in model_paths:
            print(f"  {path}")

    def combine_models(self, model_paths):
        valid_count = 0
        W_G_sum, b_G_sum, w_D_sum, b_D_sum = None, None, None, None
        for path in model_paths:
            if not os.path.exists(path):
                print(f"Model file {path} not found. Skipping.")
                continue
            data = np.load(path, allow_pickle=True)
            gen_params = data["gen_params"].item()
            disc_params = data["disc_params"].item()
            # Here, we assume that if dimensions match, the generator's output is of correct size.
            # (You might add more rigorous checks.)
            if gen_params["weights"][0].shape[1] != self.gen.weights[0].shape[1]:
                print(f"Model file {path} dimensions do not match. Skipping.")
                continue
            W_G_curr = [cp.array(W, dtype=cp.float32) for W in gen_params["weights"]]
            b_G_curr = [cp.array(b, dtype=cp.float32) for b in gen_params["biases"]]
            w_D_curr = [cp.array(W, dtype=cp.float32) for W in disc_params["weights"]]
            b_D_curr = [cp.array(b, dtype=cp.float32) for b in disc_params["biases"]]
            if W_G_sum is None:
                W_G_sum = W_G_curr
                b_G_sum = b_G_curr
                w_D_sum = w_D_curr
                b_D_sum = b_D_curr
            else:
                W_G_sum = [W_acc + W for W_acc, W in zip(W_G_sum, W_G_curr)]
                b_G_sum = [b_acc + b for b_acc, b in zip(b_G_sum, b_G_curr)]
                w_D_sum = [W_acc + W for W_acc, W in zip(w_D_sum, w_D_curr)]
                b_D_sum = [b_acc + b for b_acc, b in zip(b_D_sum, b_D_curr)]
            valid_count += 1
        if valid_count == 0:
            print("No valid model files provided for combination. Using random initialization.")
            return {"W_G": self.gen.weights, "b_G": self.gen.biases, "w_D": self.disc.weights, "b_D": self.disc.biases}
        W_G_avg = [W / valid_count for W in W_G_sum]
        b_G_avg = [b / valid_count for b in b_G_sum]
        w_D_avg = [W / valid_count for W in w_D_sum]
        b_D_avg = [b / valid_count for b in b_D_sum]
        return {"W_G": W_G_avg, "b_G": b_G_avg, "w_D": w_D_avg, "b_D": b_D_avg}
