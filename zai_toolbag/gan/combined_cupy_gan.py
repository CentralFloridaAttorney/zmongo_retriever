import os

import numpy as np
import cupy as cp

from zai_toolbag.gan import CupyGAN


class CombinedCupyGAN(CupyGAN):
    def __init__(self, image_path, model_paths, noise_dim=100, lr_D=0.001, lr_G=0.001, target_size=None):
        super().__init__(image_path, noise_dim, lr_D, lr_G, target_size=target_size)
        combined_params = self.combine_models(model_paths)
        self.W_G = combined_params["W_G"]
        self.b_G = combined_params["b_G"]
        self.w_D = combined_params["w_D"]
        self.b_D = combined_params["b_D"]
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
            data = np.load(path)
            if data["W_G"].shape[0] != self.image_size:
                print(f"Model file {path} dimensions do not match. Skipping.")
                continue
            W_G_curr = cp.array(data["W_G"], dtype=cp.float32)
            b_G_curr = cp.array(data["b_G"], dtype=cp.float32)
            w_D_curr = cp.array(data["w_D"], dtype=cp.float32)
            b_D_curr = cp.array(data["b_D"], dtype=cp.float32)
            if W_G_sum is None:
                W_G_sum = W_G_curr
                b_G_sum = b_G_curr
                w_D_sum = w_D_curr
                b_D_sum = b_D_curr
            else:
                W_G_sum += W_G_curr
                b_G_sum += b_G_curr
                w_D_sum += w_D_curr
                b_D_sum += b_D_curr
            valid_count += 1
        if valid_count == 0:
            print("No valid model files provided for combination. Using random initialization.")
            return {"W_G": self.W_G, "b_G": self.b_G, "w_D": self.w_D, "b_D": self.b_D}
        return {
            "W_G": W_G_sum / valid_count,
            "b_G": b_G_sum / valid_count,
            "w_D": w_D_sum / valid_count,
            "b_D": b_D_sum / valid_count
        }