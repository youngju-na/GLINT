#!/bin/bash

# === 사용자 설정 ===
SCENE_NAME="6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f/images"   # 예: 1234abcd5678efgh

# === 고정 설정 ===
INPUT_DIR="/home2/guest/datasets/nerfstudio/dl3dv_selected"
OUTPUT_DIR="/home2/guest/datasets/nerfstudio/dl3dv_selected_inference_output"
CONFIG_PATH="configs/rgbx_inference.yaml"
N_FRAMES=40  # 고정 chunk 크기

EXT_TYPE="JPG"  # 이미지 파일 확장자

SCENE_DIR="$INPUT_DIR/$SCENE_NAME"

# === 디렉토리 존재 확인 ===
if [ ! -d "$SCENE_DIR" ]; then
    echo "❌ Scene directory not found: $SCENE_DIR"
    exit 1
fi

# === 각 비디오 디렉토리 처리 ===
for video_dir in "$SCENE_DIR"/*; do
    if [ -d "$video_dir" ]; then
        VIDEO_NAME=$(basename "$video_dir")
        num_frames=$(ls "$video_dir"/frame*.$EXT_TYPE 2>/dev/null | wc -l)

        if [ "$num_frames" -lt "$N_FRAMES" ]; then
            echo "⚠️  Skipping $SCENE_NAME/$VIDEO_NAME — only $num_frames frames."
            continue
        fi

        echo "🚀 Inference: $SCENE_NAME/$VIDEO_NAME | Frames: $num_frames"

        python inference_svd_rgbx.py \
            --config "$CONFIG_PATH" \
            inference_input_dir="$video_dir" \
            inference_save_dir="$OUTPUT_DIR/$SCENE_NAME/$VIDEO_NAME" \
            inference_n_frames="$N_FRAMES" \
            group_mode="custom" \
            chunk_mode="first" \
            save_video_fps=6 \
            inference_res="[512,512]"
    fi
done

echo "✅ All videos in scene '$SCENE_NAME' processed."
