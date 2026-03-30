# resize_images.py
import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_images(base_dir, resized_base_dir, size=(512, 512), easyvolcap_format=False, skip_resize=False):
    for scene_name in tqdm(os.listdir(base_dir), desc="Scenes"):
        scene_path = os.path.join(base_dir, scene_name)
        if not os.path.isdir(scene_path):
            print(f"Skipping {scene_name}, not a directory.")
            continue

        input_images_dir = os.path.join(scene_path, "images")
        if not os.path.isdir(input_images_dir):
            print(f"Skipping {scene_name}, no images directory found.")
            continue

        output_images_dir = os.path.join(resized_base_dir, scene_name, "images")
        os.makedirs(output_images_dir, exist_ok=True)

        if easyvolcap_format:
            # Each subdirectory is named as a 4-digit number, containing 000000.png
            for fname in sorted(os.listdir(input_images_dir)):
                subdir = os.path.join(input_images_dir, fname)
                print(f"Processing subdirectory: {fname}")
                input_path = os.path.join(subdir, "000000.png")
                if not os.path.isfile(input_path):
                    input_path = os.path.join(subdir, "000000.jpg")
                output_fname = f"{fname.zfill(4)}.png"
                output_path = os.path.join(output_images_dir, output_fname)
            
                try:
                    img = Image.open(input_path).convert("RGB")
                    if skip_resize:
                        img.save(output_path)
                    else:
                        img = img.resize(size, Image.LANCZOS)
                        img.save(output_path)
                except Exception as e:
                    print(f"❌ Failed to process {input_path}: {e}")
        else:
            for fname in tqdm(os.listdir(input_images_dir), desc=f"Resizing images in {scene_name}"):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                input_path = os.path.join(input_images_dir, fname)
                output_path = os.path.join(output_images_dir, fname)
                try:
                    img = Image.open(input_path)
                    if img.mode == "RGBA":
                        # Separate alpha channel as mask
                        rgb_img = img.convert("RGB")
                        alpha = img.split()[-1]
                        if skip_resize:
                            # Apply mask to RGB and save as PNG with transparency
                            rgba_img = Image.merge("RGBA", (*rgb_img.split(), alpha))
                            rgba_img.save(output_path)
                        else:
                            rgb_img = rgb_img.resize(size, Image.LANCZOS)
                            alpha = alpha.resize(size, Image.NEAREST)
                            rgba_img = Image.merge("RGBA", (*rgb_img.split(), alpha))
                            rgba_img.save(output_path)
                    else:
                        img = img.convert("RGB")
                        if skip_resize:
                            img.save(output_path)
                        else:
                            img = img.resize(size, Image.LANCZOS)
                            img.save(output_path)
                except Exception as e:
                    print(f"❌ Failed to process {input_path}: {e}")

            input_path = os.path.join(input_images_dir, fname)
            output_path = os.path.join(output_images_dir, fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Original dataset base dir")
    parser.add_argument("--resized_base_dir", required=True, help="Destination base dir for resized images")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Resize dimensions (W H)")
    parser.add_argument("--easyvolcap_format", action="store_true", help="Use envs format for images")
    parser.add_argument("--skip_resize", action="store_true", help="Skip resizing and just copy images")
    args = parser.parse_args()

    resize_images(args.base_dir, args.resized_base_dir, tuple(args.size), args.easyvolcap_format, args.skip_resize)
