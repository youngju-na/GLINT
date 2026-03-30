import bpy
import numpy as np
from mathutils import Vector

# 이름 기준으로 찾음
glass_obj = bpy.data.objects.get("GlassContainer")
if glass_obj is None:
    print("GlassContainer 오브젝트를 찾을 수 없습니다.")
else:
    # glass 월드 바운딩 박스 (8코너 -> world 변환)
    world_mat = glass_obj.matrix_world
    bbox_world = [world_mat @ Vector(corner) for corner in glass_obj.bound_box]
    coords = np.array([[v.x, v.y, v.z] for v in bbox_world])
    min_corner = coords.min(axis=0)
    max_corner = coords.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    half = (max_corner - min_corner) / 2.0

    # frame 오브젝트들 찾기: 'BlackFrame' 재질을 가진 오브젝트
    frame_objs = []
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        mats = ob.data.materials
        if not mats:
            continue
        if any((m and m.name == "BlackFrame") for m in mats):
            frame_objs.append(ob)

    if not frame_objs:
        print("BlackFrame 재질을 가진 오브젝트를 찾지 못했습니다.")
    else:
        tol = 0.03  # 허용 오차 (m), 필요시 조정
        ok_count = 0
        for fo in frame_objs:
            loc = np.array(fo.matrix_world.translation)
            # 가장 가까운 점 = 각 축에서 loc을 [min_corner, max_corner]로 clamp
            closest = np.minimum(np.maximum(loc, min_corner), max_corner)
            dist = np.linalg.norm(loc - closest)
            print(f"{fo.name}: 월드 위치={loc}, closest_on_glass_bbox={closest}, 거리={dist:.4f} m")
            if dist <= tol:
                print("  -> 프레임이 유리 겉면에 가깝습니다 (OK)")
                ok_count += 1
            else:
                print("  -> 프레임이 유리에서 떨어져 있습니다 (ISSUE)")

        print(f"검사 완료: 총 프레임 {len(frame_objs)}개, 허용범위 내 {ok_count}개")

# 끝
