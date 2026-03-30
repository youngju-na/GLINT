#!/bin/bash

INPUT_DIR="/home2/guest/datasets/nerfstudio/dl3dv_selected_inference"
OUTPUT_DIR="/home2/guest/datasets/nerfstudio/dl3dv_selected_inference_output_new"
CONFIG_PATH="configs/rgbx_inference.yaml"
MIN_TAIL_FRAMES=6  # 이 수보다 작거나 같으면 이전 비디오와 합침

TMP_MERGE_DIR="/tmp/merged_inference"
mkdir -p "$TMP_MERGE_DIR"

for scene_dir in "$INPUT_DIR"/*; do
    if [ -d "$scene_dir" ]; then
        scene_name=$(basename "$scene_dir")
        video_dirs=( "$scene_dir"/* )
        num_videos=${#video_dirs[@]}

        for ((i = 0; i < num_videos; i++)); do
            video_dir="${video_dirs[$i]}"
            video_name=$(basename "$video_dir")
            num_frames=$(ls "$video_dir"/frame*.png 2>/dev/null | wc -l)

            if [ "$num_frames" -eq 0 ]; then
                echo "⚠️  Skipping $scene_name/$video_name — no frames found."
                continue
            fi

            # 너무 적은 프레임 수인 경우 이전 비디오와 병합
            if [ "$num_frames" -le "$MIN_TAIL_FRAMES" ]; then
                if [ "$i" -eq 0 ]; then
                    echo "⚠️  Only one video and too few frames. Skipping..."
                    continue
                fi

                prev_video_dir="${video_dirs[$((i-1))]}"
                prev_video_name=$(basename "$prev_video_dir")
                echo "🔀 Merging $prev_video_name and $video_name due to short length ($num_frames frames)."

                rm -rf "$TMP_MERGE_DIR"
                mkdir -p "$TMP_MERGE_DIR"

                cp "$prev_video_dir"/frame*.png "$TMP_MERGE_DIR"/
                cp "$video_dir"/frame*.png "$TMP_MERGE_DIR"/

                merged_frames=$(ls "$TMP_MERGE_DIR"/frame*.png | wc -l)
                echo "🚀 Inference: Merged $prev_video_name + $video_name | Total Frames: $merged_frames"

                python inference_svd_rgbx.py \
                    --config "$CONFIG_PATH" \
                    inference_input_dir="$TMP_MERGE_DIR" \
                    inference_save_dir="$OUTPUT_DIR/$scene_name/${prev_video_name}_merged_${video_name}" \
                    inference_n_frames="$merged_frames" \
                    model_passes="['basecolor','normal','depth','diffuse_albedo']" \
                    group_mode="custom" \
                    chunk_mode="all" \
                    save_video_fps=6 \
                    inference_res="[512,512]"

            else
                echo "🚀 Inference: $scene_name/$video_name | Frames: $num_frames"

                python inference_svd_rgbx.py \
                    --config "$CONFIG_PATH" \
                    inference_input_dir="$video_dir" \
                    inference_save_dir="$OUTPUT_DIR/$scene_name/$video_name" \
                    inference_n_frames="$num_frames" \
                    model_passes="['basecolor','normal','depth','diffuse_albedo']" \
                    group_mode="custom" \
                    chunk_mode="all" \
                    save_video_fps=6 \
                    inference_res="[512,512]"
            fi
        done
    fi
done

echo "✅ All videos processed for inference."