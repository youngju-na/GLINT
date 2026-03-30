import os
import shutil
from pathlib import Path

INPUT_DIR = "/home2/guest/datasets/nerfstudio/dl3dv_selected"
OUTPUT_DIR = "/home2/guest/datasets/nerfstudio/dl3dv_selected_inference"
CHUNK_SIZE = 60  # 최대 프레임 수

def split_scene_images(scene_path, scene_name):
    image_dir = scene_path / "images"
    if not image_dir.exists():
        print(f"❌ Skipping {scene_name} (images/ not found)")
        return

    image_list = sorted(image_dir.glob("frame_*.png"))
    total_images = len(image_list)
    num_chunks = (total_images + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total_images)

        video_name = f"video_{chunk_idx + 1}"
        target_dir = Path(OUTPUT_DIR) / scene_name / video_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, src_path in enumerate(image_list[start:end], 1):
            dst_path = target_dir / f"frame_{i:05d}.png"
            shutil.copy2(src_path, dst_path)

        print(f"✅ {scene_name}/{video_name}: {end - start} frames")

def main():
    input_root = Path(INPUT_DIR)
    for scene_path in input_root.iterdir():
        if scene_path.is_dir():
            split_scene_images(scene_path, scene_path.name)

if __name__ == "__main__":
    main()