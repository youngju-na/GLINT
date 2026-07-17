#!/usr/bin/env bash
set -euo pipefail

SCENE_ID="6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/exps/glint/ref-dl3dv/${SCENE_ID}.yaml}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/glint/ref-dl3dv/${SCENE_ID}}"
QUICK_DIR="${QUICK_DIR:-${REPO_ROOT}/results/glint/quick/${SCENE_ID}}"
MODE="${1:-quick}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ -z "${DATA_ROOT}" ]]; then
    echo "Set DATA_ROOT to the extracted scene directory." >&2
    echo "Example: DATA_ROOT=/datasets/ref-dl3dv/${SCENE_ID} $0 quick" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 2
fi
for required in \
    "${DATA_ROOT}/intri.yml" \
    "${DATA_ROOT}/extri.yml" \
    "${DATA_ROOT}/sparse/0/points3D.ply" \
    "${DATA_ROOT}/envs/points3D.ply"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing required scene input: ${required}" >&2
        exit 2
    fi
done

common=(
    python -m glint.train
    --config "${CONFIG}"
    --data-root "${DATA_ROOT}"
    --num-workers "${NUM_WORKERS:-4}"
    --max-gaussians "${MAX_GAUSSIANS:-500000}"
)

case "${MODE}" in
    quick)
        mkdir -p "${QUICK_DIR}"
        "${common[@]}" \
            --output-dir "${QUICK_DIR}" \
            --ratio 0.125 \
            --max-steps 3 \
            --transmission-start 1 \
            --reflection-start 2 \
            --max-interface-points 512 \
            --max-reflection-points 1024 \
            --no-refinement \
            --log-every 1 \
            --save-every 0 \
            --image-every 1 \
            2>&1 | tee "${QUICK_DIR}/quick.log"
        ;;
    train)
        if [[ -f "${OUTPUT_DIR}/checkpoints/latest.pt" ]]; then
            echo "Checkpoint already exists: ${OUTPUT_DIR}/checkpoints/latest.pt" >&2
            echo "Use '$0 resume' or choose another OUTPUT_DIR." >&2
            exit 2
        fi
        mkdir -p "${OUTPUT_DIR}"
        "${common[@]}" \
            --output-dir "${OUTPUT_DIR}" \
            --max-steps "${MAX_STEPS:-60000}" \
            --interface-geometry-freeze "${INTERFACE_GEOMETRY_FREEZE:-31000}" \
            --log-every "${LOG_EVERY:-10}" \
            --save-every "${SAVE_EVERY:-5000}" \
            --image-every "${IMAGE_EVERY:-1000}" \
            2>&1 | tee "${OUTPUT_DIR}/train.log"
        ;;
    resume)
        checkpoint="${OUTPUT_DIR}/checkpoints/latest.pt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "No checkpoint to resume: ${checkpoint}" >&2
            exit 2
        fi
        "${common[@]}" \
            --output-dir "${OUTPUT_DIR}" \
            --resume "${checkpoint}" \
            --max-steps "${MAX_STEPS:-60000}" \
            --interface-geometry-freeze "${INTERFACE_GEOMETRY_FREEZE:-31000}" \
            --log-every "${LOG_EVERY:-10}" \
            --save-every "${SAVE_EVERY:-5000}" \
            --image-every "${IMAGE_EVERY:-1000}" \
            2>&1 | tee -a "${OUTPUT_DIR}/train.log"
        ;;
    eval-one|eval)
        checkpoint="${OUTPUT_DIR}/checkpoints/latest.pt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "No checkpoint to evaluate: ${checkpoint}" >&2
            exit 2
        fi
        eval_args=()
        if [[ "${MODE}" == "eval-one" ]]; then
            eval_args=(--max-views "${EVAL_VIEWS:-1}")
        fi
        python -m glint.evaluate_dataset \
            --config "${CONFIG}" \
            --data-root "${DATA_ROOT}" \
            --split val \
            --checkpoint "${checkpoint}" \
            --output-dir "${OUTPUT_DIR}/${MODE}" \
            "${eval_args[@]}" \
            --save-images \
            --save-visualizations
        ;;
    *)
        echo "Usage: $0 {quick|train|resume|eval-one|eval}" >&2
        exit 2
        ;;
esac
