#!/bin/bash

# =============================================================================
# BlenderProc 3D-FRONT Rendering Script 
# Blender v3.0 사용 
# =============================================================================

SCRIPT_DIR="examples/datasets/front_3d_with_improved_mat"
RENDER_SCRIPT="$SCRIPT_DIR/load_scene_with_glass_physics.py"

# 데이터 path
FRONT_3D_FOLDER="$SCRIPT_DIR/3D-FRONT"
FUTURE_MODEL_FOLDER="$SCRIPT_DIR/3D-FUTURE-model"
TEXTURE_FOLDER="$SCRIPT_DIR/3D-FRONT-texture"
CC_MATERIALS_FOLDER="resources/cctextures/"
OUTPUT_FOLDER="$SCRIPT_DIR/renderings_trans"
OUTPUT_FOLDER_SINGLE="$SCRIPT_DIR/renderings_single_test_light_camera_centered"


SCENE_JSON="f859b03d-a55b-4e7e-8335-e02de450c873.json" # 예시로 하나의 씬만 지정

echo "[$idx/$SAMPLE_COUNT] Rendering scene: $SCENE_JSON"
blenderproc debug $RENDER_SCRIPT \
  $FRONT_3D_FOLDER \
  $FUTURE_MODEL_FOLDER \
  $TEXTURE_FOLDER \
  $SCENE_JSON \
  $CC_MATERIALS_FOLDER \
  $OUTPUT_FOLDER_SINGLE \
  --allow_on_furniture \
  --init_center_ignore_xy \
  --ignore_ceiling_in_physics