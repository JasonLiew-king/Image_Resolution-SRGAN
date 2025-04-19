import os
import argparse
from PIL import Image
from tqdm import tqdm

def upscale_image(image_path, save_path, scale=4):
    img = Image.open(image_path).convert('RGB')
    new_size = (img.width * scale, img.height * scale)
    upscaled = img.resize(new_size, Image.BICUBIC)
    upscaled.save(save_path)

def process_images(input_path, output_path, scale=4):
    if os.path.isfile(input_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        upscale_image(input_path, output_path, scale)
        print(f"Upscaled single image saved to: {output_path}")

    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        image_files = [f for f in os.listdir(input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        for filename in tqdm(image_files, desc="Upscaling images", unit="img"):
            in_file = os.path.join(input_path, filename)
            out_file = os.path.join(output_path, filename)
            upscale_image(in_file, out_file, scale)
    else:
        print("Invalid input path. Please provide a valid image file or directory.")

def main():
    parser = argparse.ArgumentParser(description="Upscale image(s) using bicubic interpolation.")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or folder")
    parser.add_argument("--output", type=str, required=True, help="Path to save upscaled image(s)")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor (default: 4)")

    args = parser.parse_args()
    process_images(args.input, args.output, args.scale)

if __name__ == "__main__":
    main()
