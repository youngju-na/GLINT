# resize_images.py
import os
from PIL import Image
from tqdm import tqdm
import argparse

def resize_images(base_dir, resized_base_dir, size=(512, 512)):
    for scene_name in tqdm(os.listdir(base_dir), desc="Scenes"):
        scene_path = os.path.join(base_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue

        input_images_dir = os.path.join(scene_path, "images")
        if not os.path.isdir(input_images_dir):
            continue

        output_images_dir = os.path.join(resized_base_dir, scene_name, "images")
        os.makedirs(output_images_dir, exist_ok=True)

        for fname in os.listdir(input_images_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(input_images_dir, fname)
            output_path = os.path.join(output_images_dir, fname)

            try:
                img = Image.open(input_path).convert("RGB")
                img = img.resize(size, Image.LANCZOS)
                img.save(output_path)
            except Exception as e:
                print(f"❌ Failed to process {input_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="Original dataset base dir")
    parser.add_argument("--resized_base_dir", required=True, help="Destination base dir for resized images")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512], help="Resize dimensions (W H)")
    args = parser.parse_args()

    resize_images(args.base_dir, args.resized_base_dir, tuple(args.size))
