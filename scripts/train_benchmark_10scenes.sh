#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF_DL3DV_ROOT="${REF_DL3DV_ROOT:-}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/glint/benchmark-10scenes}"
MODE="${1:-all}"

MAX_STEPS="${MAX_STEPS:-60000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:-500000}"
INTERFACE_GEOMETRY_FREEZE="${INTERFACE_GEOMETRY_FREEZE:-31000}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
IMAGE_EVERY="${IMAGE_EVERY:-1000}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

REF_SCENES=(
    b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d
    b9df30d6e6078880acc88acb01872c65d337f84b9dba44a23fa29c9861d7e23b
    52410f0264d14bde6acd695c637aaa274833be8afcf05ef4fd6a51176ad2dbd2
    6b42314a2f8a18a193826e2b58e45729453e74524078283f740b8f8d330c3d2f
    543b6607de9318e3a0c68b267a4b616fdc5849a140ba184807d5e70e567f8ec0
)
SYNTHETIC_SCENES=(scene_1 scene_2 scene_3 scene_4 scene_5)

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

case "${MODE}" in
    all)
        [[ -n "${REF_DL3DV_ROOT}" && -n "${SYNTHETIC_ROOT}" ]] || {
            echo "Set both REF_DL3DV_ROOT and SYNTHETIC_ROOT for mode 'all'." >&2
            exit 2
        }
        ;;
    ref)
        [[ -n "${REF_DL3DV_ROOT}" ]] || {
            echo "Set REF_DL3DV_ROOT to the directory containing ref-dl3dv scenes." >&2
            exit 2
        }
        ;;
    synthetic)
        [[ -n "${SYNTHETIC_ROOT}" ]] || {
            echo "Set SYNTHETIC_ROOT to the directory containing 3D-FRONT-T scenes." >&2
            exit 2
        }
        ;;
    *)
        echo "Usage: $0 {all|ref|synthetic}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_ROOT}"
BATCH_LOG="${OUTPUT_ROOT}/batch.log"
STATUS_FILE="${OUTPUT_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'timestamp\tdataset\tscene\tstatus\tstep\n' > "${STATUS_FILE}"
fi

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${BATCH_LOG}"
}

record_status() {
    local dataset="$1"
    local scene="$2"
    local status="$3"
    local step="$4"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "${dataset}" "${scene}" "${status}" "${step}" \
        >> "${STATUS_FILE}"
}

checkpoint_next_step() {
    python - "$1" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(payload.get("step", payload.get("epoch", -1))) + 1)
PY
}

validate_scene() {
    local config="$1"
    local data_root="$2"
    for path in \
        "${config}" \
        "${data_root}/intri.yml" \
        "${data_root}/extri.yml" \
        "${data_root}/sparse/0/points3D.ply" \
        "${data_root}/envs/points3D.ply"; do
        if [[ ! -f "${path}" ]]; then
            log "missing required input: ${path}"
            return 1
        fi
    done
}

run_scene() {
    local dataset="$1"
    local config="$2"
    local data_root="$3"
    local scene="$4"
    local output_dir="${OUTPUT_ROOT}/${dataset}/${scene}"
    local checkpoint="${output_dir}/checkpoints/latest.pt"
    local resume_args=()
    local next_step=0

    validate_scene "${config}" "${data_root}"
    mkdir -p "${output_dir}"

    if [[ -f "${checkpoint}" ]]; then
        next_step="$(checkpoint_next_step "${checkpoint}")"
        if (( next_step >= MAX_STEPS )); then
            log "skip completed ${dataset}/${scene} at step ${next_step}"
            record_status "${dataset}" "${scene}" completed "${next_step}"
            return
        fi
        resume_args=(--resume "${checkpoint}")
        log "resume ${dataset}/${scene} from step ${next_step}"
        record_status "${dataset}" "${scene}" resuming "${next_step}"
    else
        log "start ${dataset}/${scene} from initialization"
        record_status "${dataset}" "${scene}" running 0
    fi

    local command=(
        python -m glint.train
        --config "${config}"
        --data-root "${data_root}"
        --output-dir "${output_dir}"
        --max-steps "${MAX_STEPS}"
        --interface-geometry-freeze "${INTERFACE_GEOMETRY_FREEZE}"
        --num-workers "${NUM_WORKERS}"
        --max-gaussians "${MAX_GAUSSIANS}"
        --log-every "${LOG_EVERY}"
        --save-every "${SAVE_EVERY}"
        --image-every "${IMAGE_EVERY}"
        "${resume_args[@]}"
    )

    if "${command[@]}" 2>&1 | tee -a "${output_dir}/train.log" "${BATCH_LOG}"; then
        log "completed ${dataset}/${scene}"
        record_status "${dataset}" "${scene}" completed "${MAX_STEPS}"
    else
        local exit_code="${PIPESTATUS[0]}"
        log "failed ${dataset}/${scene} with exit code ${exit_code}"
        record_status "${dataset}" "${scene}" failed "${next_step}"
        return "${exit_code}"
    fi
}

if [[ -n "${WAIT_FOR_PID}" ]]; then
    log "waiting for existing training PID ${WAIT_FOR_PID}"
    while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
        sleep 30
    done
    log "existing training PID ${WAIT_FOR_PID} has exited"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "ref" ]]; then
    for scene in "${REF_SCENES[@]}"; do
        run_scene \
            ref-dl3dv \
            "${REPO_ROOT}/configs/exps/glint/ref-dl3dv/${scene}.yaml" \
            "${REF_DL3DV_ROOT}/${scene}" \
            "${scene}"
    done
fi

if [[ "${MODE}" == "all" || "${MODE}" == "synthetic" ]]; then
    for scene in "${SYNTHETIC_SCENES[@]}"; do
        run_scene \
            3d-front-t \
            "${REPO_ROOT}/configs/exps/glint/3d-front-t/${scene}.yaml" \
            "${SYNTHETIC_ROOT}/${scene}" \
            "${scene}"
    done
fi

log "all requested scenes completed"
