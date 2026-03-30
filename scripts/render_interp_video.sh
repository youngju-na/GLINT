#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Render an interpolated video between two cameras.
#
# Usage:
#   bash scripts/render_interp_video.sh \
#       --config configs/exps/envgs/ref-dl3dv/scene.yaml \
#       --exp_name envgs/ref-dl3dv/run/scene \
#       --data_root data/datasets/ref-dl3dv/scene \
#       --cam_idx1 0 --cam_idx2 16 \
#       --n_frames 120 --fps 30 \
#       --path_type arc \
#       --types RENDER,TRANSPARENCY,TRANSMISSION,REFLECTION,DIFFUSE \
#       --save_dir data/project_page
#
# Defaults:
#   --data_root is inferred as data/datasets/<dataset>/<scene>
#   --save_dir defaults to data/project_page
#   --types defaults to all project-page visualizations except ROUGHNESS
#   --cam_idx1/2 default to 0 and 16
#   --n_frames defaults to 120, --fps to 30, --path_type to arc
# ─────────────────────────────────────────────────────────────

set -e

# ── defaults ──
N_FRAMES=120
FPS=30
PATH_TYPE="arc"
CAM_IDX1=0
CAM_IDX2=16
OUTPUT=""
SAVE_DIR=""
SAVE_TAG=""
TYPES=""
CONFIG=""
EXP_NAME=""
DATA_ROOT=""
DEFAULT_DATASET_ROOT="data/datasets"
DEFAULT_SAVE_ROOT="data/project_page"
DEFAULT_TYPES="RENDER,DEPTH,NORMAL,SURFACE_NORMAL,SPECULAR,DIFFUSE,REFLECTION,TRANSPARENCY,TRANSMISSION,ENV_RENDER,TRANS_ENV_RENDER,TRANS_DEPTH,TRANS_NORMAL,OPAQUE"
VIDEO_FILTER_TYPES=("DEPTH" "NORMAL" "TRANS_DEPTH" "TRANS_NORMAL")

# ── parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)      CONFIG="$2";      shift 2 ;;
        --exp_name)    EXP_NAME="$2";    shift 2 ;;
        --data_root)   DATA_ROOT="$2";   shift 2 ;;
        --cam_idx1)    CAM_IDX1="$2";    shift 2 ;;
        --cam_idx2)    CAM_IDX2="$2";    shift 2 ;;
        --n_frames)    N_FRAMES="$2";    shift 2 ;;
        --fps)         FPS="$2";         shift 2 ;;
        --path_type)   PATH_TYPE="$2";   shift 2 ;;
        --types)       TYPES="$2";       shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --save_dir)    SAVE_DIR="$2";    shift 2 ;;
        --save_tag)    SAVE_TAG="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── infer defaults from config / exp name ──
if [ -z "$CONFIG" ] && [ -n "$EXP_NAME" ]; then
    EXP_SUFFIX="${EXP_NAME#envgs/}"
    DATASET_NAME="${EXP_SUFFIX%%/*}"
    SCENE_NAME="${EXP_NAME##*/}"
    CANDIDATE_CONFIG="configs/exps/envgs/${DATASET_NAME}/${SCENE_NAME}.yaml"
    if [ -f "$CANDIDATE_CONFIG" ]; then
        CONFIG="$CANDIDATE_CONFIG"
    fi
fi

if [ -z "$DATA_ROOT" ] && [ -n "$CONFIG" ]; then
    DATASET_NAME="$(basename "$(dirname "$CONFIG")")"
    SCENE_NAME="$(basename "$CONFIG" .yaml)"
    DATA_ROOT="${DEFAULT_DATASET_ROOT}/${DATASET_NAME}/${SCENE_NAME}"
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="${DEFAULT_SAVE_ROOT}"
fi

# ── validate ──
if [ -z "$CONFIG" ] || [ -z "$EXP_NAME" ]; then
    echo "Required: --exp_name and either --config or an exp_name that maps to configs/exps/envgs/<dataset>/<scene>.yaml"
    exit 1
fi

if [ -z "$DATA_ROOT" ]; then
    echo "Could not determine data root. Please provide --data_root or a config path that matches configs/exps/envgs/<dataset>/<scene>.yaml"
    exit 1
fi

# ── camera path output dir ──
if [ -z "$OUTPUT" ]; then
    SCENE=$(basename "$DATA_ROOT")
    OUTPUT="data/camera_paths/${SCENE}_${CAM_IDX1}_${CAM_IDX2}_${PATH_TYPE}"
fi

# ── save tag for video output ──
if [ -z "$SAVE_TAG" ]; then
    SAVE_TAG="cam${CAM_IDX1}_${CAM_IDX2}_${PATH_TYPE}"
fi

echo "═══════════════════════════════════════════════════════"
echo "  Step 1: Generating camera path (${PATH_TYPE})"
echo "═══════════════════════════════════════════════════════"
python scripts/generate_camera_path.py \
    --data_root "$DATA_ROOT" \
    --cam_idx1 "$CAM_IDX1" \
    --cam_idx2 "$CAM_IDX2" \
    --n_frames "$N_FRAMES" \
    --path_type "$PATH_TYPE" \
    --output "$OUTPUT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Step 2: Rendering with evc-test"
echo "═══════════════════════════════════════════════════════"
echo "Config: ${CONFIG}"
echo "Exp name: ${EXP_NAME}"
echo "Data root: ${DATA_ROOT}"
echo "Save root: ${SAVE_DIR}"

# Build extra args for save directory and visualization types
EXTRA_ARGS=()
EXTRA_ARGS+=("runner_cfg.visualizer_cfg.result_dir=${SAVE_DIR}")
EXTRA_ARGS+=("val_dataloader_cfg.dataset_cfg.render_path_root=${SAVE_DIR}")

if [ -z "$TYPES" ]; then
    TYPES="$DEFAULT_TYPES"
fi

TYPES_CFG="${TYPES// /}"
EXTRA_ARGS+=("runner_cfg.visualizer_cfg.types=[${TYPES_CFG}]")
echo "Selected visualization types: [${TYPES_CFG}]"

evc-test -c "${CONFIG},configs/specs/interp2cam.yaml" \
    exp_name="${EXP_NAME}" \
    val_dataloader_cfg.dataset_cfg.camera_path_intri="${OUTPUT}/intri.yml" \
    val_dataloader_cfg.dataset_cfg.camera_path_extri="${OUTPUT}/extri.yml" \
    val_dataloader_cfg.dataset_cfg.n_render_views="${N_FRAMES}" \
    runner_cfg.visualizer_cfg.video_fps="${FPS}" \
    runner_cfg.visualizer_cfg.save_tag="${SAVE_TAG}" \
    "${EXTRA_ARGS[@]}"

# ── report ──
RESULT_PATH="${SAVE_DIR}/${EXP_NAME}/${SAVE_TAG}"

apply_video_filter() {
    local type_name="$1"
    local frame_dir="${RESULT_PATH}/${type_name}"
    local output_mp4="${RESULT_PATH}/${type_name}.mp4"
    local temp_dir="${RESULT_PATH}/__filtered_video_frames__/${type_name}"

    if [ ! -d "$frame_dir" ]; then
        return
    fi

    mkdir -p "$temp_dir"

    TYPE_NAME="$type_name" FRAME_DIR="$frame_dir" TEMP_DIR="$temp_dir" python3 - <<'PY'
import os
import glob
import cv2
import numpy as np

type_name = os.environ["TYPE_NAME"]
frame_dir = os.environ["FRAME_DIR"]
temp_dir = os.environ["TEMP_DIR"]
frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))

if not frame_paths:
    raise SystemExit(0)

color_thresh = 24.0 if "DEPTH" in type_name else 36.0

for idx, frame_path in enumerate(frame_paths):
    img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    if img.ndim == 2:
        rgb = img[..., None]
        alpha = None
    elif img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3:4]
    else:
        rgb = img[..., :3]
        alpha = None

    median = np.stack([cv2.medianBlur(rgb[..., c], 3) for c in range(rgb.shape[2])], axis=-1)
    diff = np.linalg.norm(rgb.astype(np.float32) - median.astype(np.float32), axis=-1)

    if alpha is not None:
        valid = alpha[..., 0] > 8
    else:
        valid = np.any(rgb > 0, axis=-1)

    outlier = valid & (diff > color_thresh)
    filtered_rgb = rgb.copy()
    filtered_rgb[outlier] = median[outlier]

    if alpha is not None:
        filtered = np.concatenate([filtered_rgb, alpha], axis=-1)
    else:
        filtered = filtered_rgb

    cv2.imwrite(os.path.join(temp_dir, f"{idx:06d}.png"), filtered)
PY

    if ls "${temp_dir}"/*.png >/dev/null 2>&1; then
        ffmpeg -y \
            -framerate "${FPS}" \
            -i "${temp_dir}/%06d.png" \
            -c:v libx264 \
            -profile:v high \
            -level:v 4.1 \
            -tag:v avc1 \
            -pix_fmt yuv420p \
            -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            -movflags +faststart \
            "${output_mp4}" >/dev/null 2>&1
    fi
}

IFS=',' read -r -a SELECTED_TYPES <<< "$TYPES"
for type_name in "${VIDEO_FILTER_TYPES[@]}"; do
    for selected in "${SELECTED_TYPES[@]}"; do
        selected_trimmed="$(echo "$selected" | xargs)"
        if [ "$selected_trimmed" = "$type_name" ]; then
            echo "Post-processing ${type_name}.mp4 with outlier suppression"
            apply_video_filter "$type_name"
            break
        fi
    done
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done! Videos saved to:"
echo "  ${RESULT_PATH}/"
echo "═══════════════════════════════════════════════════════"
