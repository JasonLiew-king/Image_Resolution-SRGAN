import os
import argparse
from PIL import Image
from tqdm import tqdm  # Progress bar

def downscale_images(input_dir, output_dir, scale_factor=4):
    """ Converts HR images to LR images by downscaling using bicubic interpolation. """
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Processing {len(image_files)} images from {input_dir}...")
    
    for image_file in tqdm(image_files):
        img_path = os.path.join(input_dir, image_file)
        img = Image.open(img_path)

        # Resize (downscale)
        lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)

        # Save LR image
        lr_img.save(os.path.join(output_dir, image_file))

    print(f"Done! LR images saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HR images to LR images with x4 downscaling.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the HR image folder (train/valid)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save LR images")
    
    args = parser.parse_args()
    downscale_images(args.input_dir, args.output_dir)
