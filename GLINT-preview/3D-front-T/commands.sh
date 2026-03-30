#!/bin/bash

# =============================================================================
# BlenderProc 3D-FRONT Rendering Script
# =============================================================================

# 경로 설정 (필요에 따라 수정하세요)
SCRIPT_DIR="examples/datasets/front_3d_with_improved_mat"

# RENDER_SCRIPT="$SCRIPT_DIR/render_dataset_improved_mat_transparency_gemini.py"
# RENDER_SCRIPT_GPT="$SCRIPT_DIR/render_dataset_improved_mat_single.py.py"
RENDER_SCRIPT="$SCRIPT_DIR/render_dataset_improved_mat_single.py"
# RENDER_SCRIPT="$SCRIPT_DIR/render_dataset_improved_mat_single_camera_centered.py"
MULTI_RENDER_SCRIPT="$SCRIPT_DIR/multi_render.py"

# 데이터 경로
FRONT_3D_FOLDER="$SCRIPT_DIR/3D-FRONT"
FUTURE_MODEL_FOLDER="$SCRIPT_DIR/3D-FUTURE-model"
TEXTURE_FOLDER="$SCRIPT_DIR/3D-FRONT-texture"
CC_MATERIALS_FOLDER="resources/cctextures/"
OUTPUT_FOLDER="$SCRIPT_DIR/renderings_trans"
OUTPUT_FOLDER_SINGLE="$SCRIPT_DIR/renderings_single_test_light_camera_centered"

# =============================================================================
# 여러 씬 랜덤 샘플 렌더링
# =============================================================================
NUM_SCENES=30

# FRONT_3D_FOLDER 바로 아래(.json)에서만 추출해 basename 목록 생성
mapfile -t ALL_JSONS < <(find "$FRONT_3D_FOLDER" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort)

if [ ${#ALL_JSONS[@]} -eq 0 ]; then
  echo "No JSON files found in $FRONT_3D_FOLDER" >&2
  exit 1
fi

SAMPLE_COUNT=$NUM_SCENES
if [ ${#ALL_JSONS[@]} -lt $NUM_SCENES ]; then
  SAMPLE_COUNT=${#ALL_JSONS[@]}
fi

mapfile -t SCENE_LIST < <(printf '%s\n' "${ALL_JSONS[@]}" | shuf -n "$SAMPLE_COUNT")

echo "Rendering $SAMPLE_COUNT random scenes from $FRONT_3D_FOLDER"

idx=1
for SCENE_JSON in "${SCENE_LIST[@]}"; do
  echo "[$idx/$SAMPLE_COUNT] Rendering scene: $SCENE_JSON"
  blenderproc run $RENDER_SCRIPT \
    $FRONT_3D_FOLDER \
    $FUTURE_MODEL_FOLDER \
    $TEXTURE_FOLDER \
    $SCENE_JSON \
    $CC_MATERIALS_FOLDER \
    $OUTPUT_FOLDER_SINGLE \
    --allow_on_furniture \
    --init_center_ignore_xy \
    --ignore_ceiling_in_physics
  idx=$((idx+1))
  echo "---"
done
