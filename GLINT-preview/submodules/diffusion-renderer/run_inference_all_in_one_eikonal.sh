#!/bin/bash

# ====== 기본 경로 설정 ======
ORIGINAL_BASE_DIR="/mnt/nas3/youngju/glint/nerfstudio/eikonal"
RESIZED_BASE_DIR="/mnt/nas3/youngju/glint/nerfstudio/eikonal"
INFERENCE_BASE_SAVE_DIR="/mnt/nas3/youngju/glint/nerfstudio/eikonal_diffrens"

# EXCLUDE_SCENES=("scene_1" "scene_2" "scene_3" "scene_4" "scene_5" "scene_6" "scene_7" "scene_8")  # 제외할 scene 이름들을 여기에 추가


# 사용할 GPU ID들
GPUS=(0)
NUM_GPUS=${#GPUS[@]}
GPU_INDEX=0

# echo "===================================================="
# echo "🛠️  Step 1: Resizing all images to 512x512"
# echo "===================================================="

# python resize_images.py \
#     --base_dir "${ORIGINAL_BASE_DIR}" \
#     --resized_base_dir "${RESIZED_BASE_DIR}" \
#     --size 512 512

# if [ $? -ne 0 ]; then
#     echo "❌ Resizing failed. Aborting inference."
#     exit 1
# fi

echo ""
echo "===================================================="
echo "🚀 Step 2: Launching parallel inference on GPUs"
echo "===================================================="

BASE_DIR="${RESIZED_BASE_DIR}"
# BASE_DIR="${ORIGINAL_BASE_DIR}"

declare -a PIDS

find "${BASE_DIR}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | while IFS= read -r scene_name; do

    skip_scene=false
    for exclude in "${EXCLUDE_SCENES[@]}"; do
        if [ "${scene_name}" == "${exclude}" ]; then
            echo "Skipping excluded scene: ${scene_name}"
            skip_scene=true
            break
        fi
    done
    if [ "$skip_scene" = true ]; then
        continue
    fi

    CURRENT_INFERENCE_INPUT_DIR="${BASE_DIR}/${scene_name}/images"
    CURRENT_INFERENCE_SAVE_DIR="${INFERENCE_BASE_SAVE_DIR}/${scene_name}"
    CURRENT_GPU=${GPUS[$GPU_INDEX]}

    echo "Launching ${scene_name} on GPU ${CURRENT_GPU}"

    CUDA_VISIBLE_DEVICES=${CURRENT_GPU} python inference_svd_rgbx.py \
            --config configs/rgbx_inference.yaml \
            inference_input_dir="${CURRENT_INFERENCE_INPUT_DIR}" \
            inference_save_dir="${CURRENT_INFERENCE_SAVE_DIR}" \
            inference_n_frames=20 \
            inference_n_steps=20 \
            inference_res="[512,512]" \
            model_passes="['basecolor','normal','depth','diffuse_albedo']" \
            chunk_mode="all" &

    PIDS+=($!)
    GPU_INDEX=$(( (GPU_INDEX + 1) % NUM_GPUS ))

    if [ ${#PIDS[@]} -ge $NUM_GPUS ]; then
        wait "${PIDS[@]}"
        PIDS=()
    fi
done

wait "${PIDS[@]}"
echo "✅ All eligible scene inferences completed."
