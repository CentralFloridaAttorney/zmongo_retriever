import argparse
import os

from zai_toolbag.gan import get_next_model_filename, create_sample_image, CupyGAN, CombinedCupyGAN


# ----------------------------
# Main training loop with command-line options
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a customizable CuPy GAN on an image.")
    parser.add_argument("--sample", type=str, default="red_circle",
                        choices=["red_circle", "blue_square", "blue_circle"],
                        help="Type of sample image to create if image file is not found (default: red_circle)")
    parser.add_argument("--image", type=str, default="/media/overlordx/DATA/resources/images/generated/ComfyUI_00118_.png",
                        help="Path to the training image (default depends on --sample)")
    parser.add_argument("--model", type=str, default="model_checkpoints/gan_model_red_circle_v013.npz",
                        help="Base path for saved model parameters (a new version will be created)")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Number of training epochs")
    parser.add_argument("--log_interval", type=int, default=20000,
                        help="Interval (in epochs) to save output images and model")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for both generator and discriminator")
    parser.add_argument("--combine_models", type=str, default="",
                        help="Comma-separated list of model files to combine before training")
    parser.add_argument("--target_size", type=str, default="128x128",
                        help="Target size in WxH format (default: 128x128)")
    args = parser.parse_args()

    try:
        width, height = map(int, args.target_size.lower().split("x"))
        target_size = (width, height)
    except Exception as e:
        print(f"Error parsing target_size: {e}")
        exit(1)

    create_sample_image(args.sample, args.image, size=target_size)

    if args.combine_models:
        model_paths = [p.strip() for p in args.combine_models.split(",") if p.strip()]
        gan = CombinedCupyGAN(image_path=args.image, model_paths=model_paths,
                              noise_dim=100, lr_D=args.lr, lr_G=args.lr, target_size=target_size)
    else:
        gan = CupyGAN(image_path=args.image, noise_dim=100, lr_D=args.lr, lr_G=args.lr, target_size=target_size)

    output_dir = "gan_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_dir = "model_checkpoints"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if os.path.exists(args.model):
        gan.load_model(args.model)
    else:
        print("No saved model found. Starting from scratch.")

    for epoch in range(args.epochs):
        loss_D, loss_G = gan.train_step()
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch}: D_loss = {loss_D:.4f}, G_loss = {loss_G:.4f}")
            output_image = gan.generate_image()
            output_file = os.path.join(output_dir, f"output_epoch_{epoch}.jpg")
            output_image.save(output_file)
            print(f"Saved output image to {output_file}")
            new_model_file = get_next_model_filename(args.model, model_dir)
            gan.save_model(new_model_file)
            args.model = new_model_file

    print("Training complete.")


if __name__ == "__main__":
    main()
