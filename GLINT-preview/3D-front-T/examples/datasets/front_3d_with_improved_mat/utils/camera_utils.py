import numpy as np
import blenderproc as bproc

def _look_at(cam_pos, target, up=np.array([0,0,1.0])):
    f = (target - cam_pos); f = f / (np.linalg.norm(f) + 1e-8)
    r = np.cross(up, f);   r = r / (np.linalg.norm(r) + 1e-8)
    u = np.cross(f, r)
    T = np.eye(4)
    T[:3,0] = r; T[:3,1] = u; T[:3,2] = -f; T[:3,3] = cam_pos
    return T

def add_video_trajectory_around_glass(
    center,                     # glass center (np.array([x,y,z]))
    room_min, room_max,         # room AABB (np.array)
    bvh_tree,                   # for obstacle check
    seconds=8.0, fps=24,        # video 길이
    r_min=1.2, r_max=2.4,       # 반경 범위(가까이/멀리 왕복)
    heights=(1.1, 1.5),         # 높이 범위(천천히 오르내림)
    laps=1.0,                   # 몇 바퀴 도는지
    dolly_freq=1.0,             # 반경 펌핑 빈도(왕복 횟수)
    yaw_offset=0.0,             # 전체 회전 오프셋(rad)
    jitter=0.01,                # 미세 손떨림(너무 크면 어지러움)
    ensure_visibility=True
):
    """
    방 전체를 부드럽게 돌며(center를 계속 주시) 포즈들을 추가.
    반환: 추가된 프레임 수
    """
    N = int(seconds * fps)
    t = np.linspace(0.0, 1.0, N, endpoint=True)

    # 반경과 높이를 부드럽게 변화(도넛 궤도 + 살짝 도리질)
    r   = 0.5*(r_min + r_max) + 0.5*(r_max - r_min)*np.sin(2*np.pi*dolly_freq*t)
    h   = 0.5*(heights[0] + heights[1]) + 0.5*(heights[1]-heights[0])*np.cos(2*np.pi*t)
    az  = yaw_offset + 2*np.pi*laps*t   # 균일한 각속도

    added = 0
    for i in range(N):
        pos = center + np.array([r[i]*np.cos(az[i]), r[i]*np.sin(az[i]), h[i]])

        # 방 AABB 안쪽만 허용
        if not (room_min[0] < pos[0] < room_max[0] and room_min[1] < pos[1] < room_max[1]):
            continue

        # 미세한 핸드헬드 느낌(논문용이면 0.0~0.01 정도 권장)
        if jitter > 0:
            pos = pos + np.random.normal(scale=jitter, size=3) * np.array([1,1,0.3])

        cam2world = _look_at(pos, center)

        # 시야 장애물 검사(가구/벽으로 완전 가려지면 스킵)
        if ensure_visibility:
            if not bproc.camera.perform_obstacle_in_view_check(cam2world, {}, bvh_tree):
                continue

        bproc.camera.add_camera_pose(cam2world)
        added += 1

    return added