import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # for loading and resizing images

class SimpleToyGAN:
    """
    A simple toy implementation of a Generative Adversarial Network (GAN)
    for generating 2x2 black and white images. Now enhanced to optionally
    load a real image file as training data and to specify epochs dynamically.
    """

    def __init__(self,
                 real_samples=None,
                 image_path=None,
                 new_shape=(2, 2),
                 default_backslash=True,
                 learning_rate=0.01,
                 epochs=5000,
                 seed=42):
        """
        Constructor for the SimpleToyGAN class.

        Parameters
        ----------
        real_samples : list of numpy arrays, optional
            Each numpy array should be shape (4,) or (2,2). These are the real images.
        image_path : str, optional
            Path to an image file to load as a real sample.
        new_shape : tuple of int
            The target size to which we resize the loaded image. Default is (2,2).
        default_backslash : bool
            Whether to include the default 'backslash' samples. If you only want to train
            on your own image, set this to False.
        learning_rate : float
            The learning rate used for both the Discriminator and Generator.
        epochs : int
            Default number of training epochs. Can be overridden in the 'train' method.
        seed : int
            Random seed for reproducibility.
        """

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Default real_samples
        if real_samples is None:
            real_samples = []
        self.real_samples = real_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

        # If requested, add default "backslash" samples
        if default_backslash:
            self._append_default_backslash_samples()

        # If an image path is provided, load that image and add as real sample
        if image_path:
            img_array = self._load_image_as_array(image_path, new_shape)
            self.real_samples.append(img_array)

        # Instantiate Discriminator and Generator
        self.D = self.Discriminator()
        self.G = self.Generator()

        # For tracking training error
        self.err_D = []
        self.err_G = []

    def _append_default_backslash_samples(self):
        """
        Append the default 2x2 'backslash' samples to self.real_samples.
        """
        default_samples = [
            np.array([1.0, 0.0, 0.0, 1.0]),
            np.array([0.9, 0.1, 0.2, 0.8]),
            np.array([0.9, 0.2, 0.1, 0.8]),
            np.array([0.8, 0.1, 0.2, 0.9]),
            np.array([0.8, 0.2, 0.1, 0.9])
        ]
        self.real_samples.extend(default_samples)

    def _load_image_as_array(self, image_path, new_shape=(2, 2)):
        """
        Loads and preprocesses an image file as a flattened NumPy array.
        Resizes to new_shape, converts to grayscale, and scales pixel values to [0, 1].
        Returns a 1D array of shape (new_shape[0]*new_shape[1],).
        """
        # Open and convert image to grayscale
        with Image.open(image_path).convert("L") as img:
            # Resize
            img_resized = img.resize(new_shape)
            # Convert to NumPy array
            img_array = np.array(img_resized, dtype=np.float32)
            # Scale pixel intensities to [0, 1]
            img_array /= 255.0
            # Flatten to shape (4,) if new_shape=(2,2)
            img_array = img_array.flatten()
            return img_array

    class Discriminator:
        """
        The Discriminator learns to classify real vs. generated (fake) images.
        """

        def __init__(self):
            # Randomly initialize weights and bias
            self.weights = np.array([np.random.normal() for _ in range(4)])
            self.bias = np.random.normal()

        def sigmoid(self, x):
            return np.exp(x) / (1.0 + np.exp(x))

        def forward(self, X):
            return self.sigmoid(np.dot(X, self.weights) + self.bias)

        def error_from_image(self, image):
            """
            For real images, label is 1. Error = -ln(D(x)).
            """
            pred = self.forward(image)
            return -np.log(pred + 1e-12)  # Add small epsilon for numerical stability

        def deriv_from_images(self, image):
            """
            Compute derivatives w.r.t. weights and bias for a real image (label=1).
            """
            dx = self.forward(image)
            # dE/dW = -(1 - D(x)) * x
            d_wts = -(1.0 - dx) * image
            # dE/dBias = -(1 - D(x))
            d_bias = -(1.0 - dx)
            return d_wts, d_bias

        def update_from_image(self, x, lr):
            """
            Backprop for a real image (label=1).
            """
            d_wts, d_bias = self.deriv_from_images(x)
            self.weights -= lr * d_wts
            self.bias -= lr * d_bias

        def error_from_noise(self, noise):
            """
            For noise/fake images, label is 0. Error = -ln(1 - D(x)).
            """
            pred = self.forward(noise)
            return -np.log((1.0 - pred) + 1e-12)

        def deriv_from_noise(self, noise):
            """
            Compute derivatives w.r.t. weights and bias for a fake image (label=0).
            """
            dx = self.forward(noise)
            # dE/dW = D(x) * x
            d_wts = dx * noise
            # dE/dBias = D(x)
            d_bias = dx
            return d_wts, d_bias

        def update_from_noise(self, noise, lr):
            """
            Backprop for a fake image (label=0).
            """
            d_wts, d_bias = self.deriv_from_noise(noise)
            self.weights -= lr * d_wts
            self.bias -= lr * d_bias

    class Generator:
        """
        The Generator class used in the SimpleToyGAN.
        Learns to generate fake images that fool the Discriminator.
        """

        def __init__(self):
            # Randomly initialize weights and biases (each a length-4 vector)
            self.weights = np.array([np.random.normal() for _ in range(4)])
            self.bias = np.array([np.random.normal() for _ in range(4)])

        def sigmoid(self, x):
            return np.exp(x) / (1.0 + np.exp(x))

        def forward(self, z):
            """
            Given a scalar z in [0,1], produce a 4-d vector (2x2 image).
            """
            return self.sigmoid(z * self.weights + self.bias)

        def error(self, z, D):
            """
            Generator wants D(G(z)) = 1 => E_G = -ln(D(G(z))).
            """
            x = self.forward(z)        # generated image
            y = D.forward(x)           # D(G(z))
            return -np.log(y + 1e-12)

        def deriv(self, z, D):
            """
            Derivatives of the Generator's parameters w.r.t. the loss -ln(D(G(z))).
            """
            x = self.forward(z)        # G(z)
            y = D.forward(x)           # D(G(z))

            # For the Generator, derivative chain from -ln(y):
            # d/dw [ -ln(D(G(z))) ] = -(1 - y)*dW_D(...) * partial_of_generator
            # We approximate partial_of_generator via standard chain rule on x = sigmoid(...)
            factor = -(1.0 - y) * D.weights * x * (1.0 - x)
            dwts_g = factor * z
            dbs_g = factor

            return dwts_g, dbs_g

        def update(self, z, D, lr):
            """
            Update Generator parameters using gradient from fake images.
            """
            dw, db = self.deriv(z, D)
            self.weights -= lr * dw
            self.bias -= lr * db

    def train(self, epochs=None):
        """
        Train the GAN on the real samples for the specified number of epochs.
        If epochs is not provided, uses self.epochs (set in __init__).
        """
        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            for real_img in self.real_samples:
                # 1. Update D with real image
                self.D.update_from_image(real_img, lr=self.learning_rate)

                # 2. Generate random scalar z
                z = np.random.rand()

                # 3. Discriminator error:
                #    E_D = -ln(D(real_img)) - ln(1 - D(G(z)))
                fake_img = self.G.forward(z)
                e_d_real = self.D.error_from_image(real_img)
                e_d_fake = self.D.error_from_noise(fake_img)
                e_d = e_d_real + e_d_fake
                self.err_D.append(e_d)

                # 4. Generator error: E_G = -ln(D(G(z)))
                e_g = self.G.error(z, self.D)
                self.err_G.append(e_g)

                # 5. Update D with fake image (label=0)
                self.D.update_from_noise(fake_img, lr=self.learning_rate)

                # 6. Update G using gradient from D
                self.G.update(z, self.D, lr=self.learning_rate)

    def generate_images(self, num_images=4):
        """
        Generate 'num_images' fake images from the trained Generator.
        Returns a list of numpy arrays (each shape (4,)).
        """
        generated = []
        for _ in range(num_images):
            z = np.random.rand()
            generated_image = self.G.forward(z)
            generated.append(generated_image)
        return generated

    def plot_generated_images(self, num_images=4):
        """
        Generate and plot some images from the trained Generator.
        """
        images = self.generate_images(num_images)
        self.view_images(images, rows=1, cols=num_images)
        print("Sample generated image arrays:")
        for img in images:
            print(img.reshape(2,2), "\n")

    def view_images(self, matrices, rows, cols):
        """
        Utility function to visualize a list of 2x2 images in a grid.
        """
        fig, axes = plt.subplots(figsize=(10,10), nrows=rows, ncols=cols, sharex=True, sharey=True)
        # Ensure axes is a 2D array even for single row
        axes = np.array(axes).reshape(rows, cols)
        for axis, image in zip(axes.flatten(), matrices):
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            # Expecting image of shape (4,) => reshape to (2,2)
            reshaped_img = image.reshape((2,2))
            # We do (1 - image) to invert color intensities for better visual contrast
            axis.imshow(1.0 - reshaped_img, cmap='Greys_r')
        plt.show()

    def plot_errors(self):
        """
        Plot the error curves for both Discriminator and Generator over training.
        """
        plt.figure(figsize=(10,5))
        plt.plot(self.err_D, color='blue', label='Discriminator Error')
        plt.plot(self.err_G, color='red',  label='Generator Error')
        plt.xlabel("Training Steps")
        plt.ylabel("Error")
        plt.legend()
        plt.title("Error vs. Training Steps")
        plt.show()

    def print_final_parameters(self):
        """
        Print out the final learned parameters of the Discriminator and Generator.
        """
        print("DISCRIMINATOR PARAMETERS:")
        print("Weights:", self.D.weights)
        print("Bias:", self.D.bias, "\n")

        print("GENERATOR PARAMETERS:")
        print("Weights:", self.G.weights)
        print("Biases:", self.G.bias)
        print()


# ------------------------
# Example usage:
# ------------------------
if __name__ == "__main__":
    """
    Example of using the improved SimpleToyGAN:

    1. If you have a single real image you want to train on (converted to 2x2),
       specify `image_path='path/to/image.png'` and set `default_backslash=False`
       if you don't want the default backslash images.

    2. If you want to train for a custom number of epochs, pass that to the `train` method.
    """
    # Example: Use a random image from disk (NOTE: must be small or scaled to 2x2 effectively)
    gan = SimpleToyGAN(image_path="/home/overlordx/resources/panos/IMG_20250303_061902_045.jpg",
                       new_shape=(2,2),
                       default_backslash=False,
                       learning_rate=0.01,
                       epochs=1_000,
                       seed=45)

    # For demonstration, let's skip an actual file path and just do the defaults:
    # gan = SimpleToyGAN(learning_rate=0.01, epochs=1000, seed=45)

    # Train for a custom number of epochs (overrides constructor):
    gan.train(epochs=2000)

    # Generate and plot images
    gan.plot_generated_images(num_images=4)

    # Plot error curves
    gan.plot_errors()

    # Print final parameters
    gan.print_final_parameters()
