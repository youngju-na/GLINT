#!/bin/bash

# 기본 경로 설정
COLMAP_ROOT="/home/youngju/ssd/datasets/nerfstudio/dl3dv_selected"
EASYVOLCAP_ROOT="/home/youngju/ssd/datasets/nerfstudio/dl3dv_selected_envgs"
SCRIPT_PATH="scripts/preprocess/colmap_to_easyvolcap.py"

# 출력 디렉토리 생성
mkdir -p "$EASYVOLCAP_ROOT"

# COLMAP_ROOT 내의 모든 디렉토리(scene)에 대해 반복
for scene_path in "$COLMAP_ROOT"/*; do
  if [ -d "$scene_path" ]; then
    scene_name=$(basename "$scene_path")
    echo "Processing scene: $scene_name"

    # EnvGS 포맷으로 변환하는 Python 스크립트 실행
    python "$SCRIPT_PATH" \
      --data_root "$COLMAP_ROOT" \
      --output "$EASYVOLCAP_ROOT/$scene_name" \
      --scenes "$scene_name" \
      --colmap sparse/0
  fi
done

echo "All scenes have been processed."