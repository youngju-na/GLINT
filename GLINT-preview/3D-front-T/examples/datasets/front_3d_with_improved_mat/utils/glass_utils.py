# glass_synth_utils.py
"""
Utilities to synthesize two transparency cases in Front3D renders:
  (1) Glass container with a floating opaque object at the room center
  (2) Window cut into a wall (central 1/4 area), with frame + glass pane

Designed as drop-in helpers for render_dataset_improved_mat.py.

Usage (high level):
    from glass_synth_utils import (
        GlassParams, WindowParams,
        make_glass_material, prune_to_single_room,
        create_glass_container_with_object,
        cut_window_and_insert_glass,
        boost_interest_scores_for_transparency
    )

Then, **after** loading Front3D objects and sampling/applying base materials,
call in this order (if you wish):
    room_id = prune_to_single_room(loaded_objects)
    if args.use_glass_container:
        create_glass_container_with_object(loaded_objects, room_id, args.asset_dir, GlassParams(...))
    if args.use_glass_window:
        cut_window_and_insert_glass(loaded_objects, room_id, WindowParams(...))
    boost_interest_scores_for_transparency(interest_score_setting)

Notes:
- Requires BlenderProc to run inside Blender (bpy available). Boolean ops are handled via bpy modifiers.
- If boolean fails (non-manifold/concave mesh), we fall back to an overlay window (no hole, just framed glass slightly in front of wall).
- All added objects receive cp tags for easy segmentation: cp_inst_mark, cp_uid, cp_room_id.
"""
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import blenderproc as bproc

from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.loader.BlendLoader import load_blend
from pathlib import Path

# --- Parameters --------------------------------------------------------------

@dataclass
class GlassParams:
    container_outer_size: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # meters (X,Y,Z)
    wall_thickness: float = 0.02  # meters per side
    ior: float = 1.45
    roughness: float = 0.02
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    inner_object_float_height: float = 0.2  # meters above container center
    margin_ratio: float = 0.85  # inner object max fraction of inner extents

@dataclass
class WindowParams:
    rel_width: float = 0.5   # central width fraction of chosen wall
    rel_height: float = 0.5  # central height fraction
    frame_thickness: float = 0.05  # meters
    glass_thickness: float = 0.01  # meters
    ior: float = 1.45
    roughness: float = 0.02
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    cut_method: str = "boolean"  # {"boolean", "overlay"}

# --- Materials --------------------------------------------------------------

def make_glass_material(name: str = "glass_clear", ior: float = 1.45, roughness: float = 0.02,
                        color=(1.0, 1.0, 1.0)) -> bproc.types.Material:
    mat = bproc.material.create(name)
    # Principled BSDF
    mat.set_principled_bsdf_value("Base Color", list(color))
    mat.set_principled_bsdf_value("Transmission", 1.0)
    mat.set_principled_bsdf_value("Roughness", float(roughness))
    mat.set_principled_bsdf_value("IOR", float(ior))
    mat.set_principled_bsdf_value("Specular", 0.5)
    # Optional: thin surface look
    mat.set_principled_bsdf_value("Alpha", 1.0)
    return mat


def make_opaque_material(name: str = "opaque_default", color=(0.8, 0.5, 0.2), roughness: float = 0.5) -> bproc.types.Material:
    mat = bproc.material.create(name)
    mat.set_principled_bsdf_value("Base Color", list(color))
    mat.set_principled_bsdf_value("Transmission", 0.0)
    mat.set_principled_bsdf_value("Roughness", float(roughness))
    mat.set_principled_bsdf_value("Specular", 0.5)
    return mat

# --- Room utilities ---------------------------------------------------------

def _obj_world_bbox(obj: bproc.types.MeshObject) -> np.ndarray:
    """Return 8x3 bbox corners in world coordinates."""
    bb = np.array(obj.get_bound_box())  # BlenderProc returns world-space corners
    return bb


def _bbox_center_and_size(bb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = bb.mean(axis=0)
    size = bb.max(axis=0) - bb.min(axis=0)
    return center, size


def prune_to_single_room(loaded_objects: List[bproc.types.MeshObject]) -> Optional[str]:
    """Keep only the largest-floor-area room. Returns selected cp_room_id (str).
    Removes all objects whose cp_room_id != selected.
    If cp_room_id missing, we keep the object (global things e.g., ceiling lights) to avoid breaking the scene.
    """
    floors = [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject) and o.get_name().startswith("Floor")]
    if not floors:
        return None

    # choose the floor with the largest XY bbox area
    best = None
    best_area = -1.0
    for f in floors:
        bb = _obj_world_bbox(f)
        # floors are mostly horizontal; project on XY
        size = bb.max(axis=0) - bb.min(axis=0)
        area = float(size[0] * size[1])
        rid = f.get_cp("room_id")  # -> cp_room_id
        if area > best_area and rid is not None:
            best_area = area
            best = (rid, f)
    if best is None:
        return None
    keep_room_id = best[0]

    # Delete objects strictly belonging to other rooms
    to_delete = []
    for o in loaded_objects:
        rid = o.get_cp("room_id")
        if rid is not None and rid != keep_room_id:
            to_delete.append(o)
    for o in to_delete:
        try:
            o.delete()
        except Exception:
            pass

    # Stamp room id on any new assets we create later
    for o in loaded_objects:
        if o.get_cp("room_id") is None:
            o.set_cp("room_id", keep_room_id)

    return keep_room_id

# --- Asset loading ----------------------------------------------------------

def _pick_asset(path_dir: str) -> Optional[str]:
    if not path_dir or not os.path.isdir(path_dir):
        return None
    cand = [f for f in os.listdir(path_dir) if f.lower().endswith((".obj", ".glb", ".fbx", ".blend"))]
    return None if not cand else os.path.join(path_dir, random.choice(cand))


def _load_single_object(asset_path: str) -> List[bproc.types.MeshObject]:
    ext = os.path.splitext(asset_path)[1].lower()
    if ext == ".obj":
        return bproc.loader.load_obj(asset_path)
    elif ext == ".glb":
        return bproc.loader.load_glb(asset_path)
    elif ext == ".fbx":
        return bproc.loader.load_fbx(asset_path)
    elif ext == ".blend":
        # Load all mesh objects from the blend file root collection
        return bproc.loader.load_blend(asset_path)
    else:
        raise ValueError(f"Unsupported asset type: {ext}")

# --- Glass container with floating opaque object ---------------------------

def create_glass_container_with_object(loaded_objects: List[bproc.types.MeshObject], room_id: Optional[str],
                                       asset_dir: Optional[str], params: GlassParams) -> Tuple[bproc.types.MeshObject, Optional[bproc.types.MeshObject]]:
    # Find room center from the largest floor
    floors = [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject) and o.get_name().startswith("Floor")
              and (room_id is None or o.get_cp("room_id") == room_id)]
    if not floors:
        raise RuntimeError("No floor found to place glass container.")
    bb = _obj_world_bbox(floors[0])
    room_center, room_size = _bbox_center_and_size(bb)

    # Outer container (cube shell)
    outer = bproc.object.create_primitive("cube", scale=[s * 0.5 for s in params.container_outer_size])
    outer.set_location(room_center)
    outer.set_cp("inst_mark", "glass_container")
    if room_id is not None:
        outer.set_cp("room_id", room_id)
    outer.set_name("GlassContainerOuter")

    glass_mat = make_glass_material("glass_container_mat", params.ior, params.roughness, params.color)
    # make walls by using solid cube and setting it to glass; visually fine for thin walls
    for i in range(len(outer.get_materials())):
        outer.set_material(i, glass_mat)

    # Inner object (opaque), floating at center + height
    inner_obj = None
    asset_path = _pick_asset(asset_dir) if asset_dir else None
    if asset_path is not None:
        objs = _load_single_object(asset_path)
        if objs:
            inner_obj = objs[0]
    else:
        # fallback: procedural sphere
        inner_obj = bproc.object.create_primitive("uv_sphere", radius=0.2)

    inner_obj.set_name("ContainerInnerOpaque")
    inner_obj.set_cp("inst_mark", "opaque_inside")
    if room_id is not None:
        inner_obj.set_cp("room_id", room_id)

    opaque_mat = make_opaque_material("opaque_inner")
    for i in range(len(inner_obj.get_materials())):
        inner_obj.set_material(i, opaque_mat)

    # Fit inner object into container with a margin
    # Compute inner extents (outer size minus 2*wall)
    inner_extents = np.array(params.container_outer_size) - 2.0 * params.wall_thickness
    inner_extents = np.maximum(inner_extents, 1e-3)

    # Compute object bbox size and scale
    bb_in = _obj_world_bbox(inner_obj)
    size_in = bb_in.max(axis=0) - bb_in.min(axis=0)
    size_in[size_in < 1e-6] = 1e-6
    target = params.margin_ratio * inner_extents
    scale_factor = float(np.min(target / size_in))
    inner_obj.set_scale(list((np.array(inner_obj.get_scale()) * scale_factor)))

    # Place at container center + height
    new_loc = room_center.copy()
    new_loc[2] += params.inner_object_float_height
    inner_obj.set_location(new_loc)

    return outer, inner_obj

# --- Window creation (boolean or overlay) -----------------------------------

def _choose_wall_in_room(loaded_objects: List[bproc.types.MeshObject], room_id: Optional[str]) -> Optional[bproc.types.MeshObject]:
    candidates = []
    for o in loaded_objects:
        if not isinstance(o, bproc.types.MeshObject):
            continue
        if not o.get_name().startswith("Wall"):
            continue
        rid = o.get_cp("room_id")
        if room_id is not None and rid != room_id:
            continue
        # Prefer larger walls (by bbox area of the two largest axes)
        bb = _obj_world_bbox(o)
        size = bb.max(axis=0) - bb.min(axis=0)
        area = np.product(np.sort(size)[-2:])
        candidates.append((area, o))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def cut_window_and_insert_glass(loaded_objects: List[bproc.types.MeshObject], room_id: Optional[str],
                                params: WindowParams) -> Tuple[bproc.types.MeshObject, bproc.types.MeshObject, Optional[bproc.types.MeshObject]]:
    import bpy  # only valid inside Blender/BlenderProc

    wall = _choose_wall_in_room(loaded_objects, room_id)
    if wall is None:
        raise RuntimeError("No wall found to cut a window.")

    wall_bb = _obj_world_bbox(wall)
    wall_center, wall_size = _bbox_center_and_size(wall_bb)
    # Window opening size = central rel_width x rel_height of wall bbox (on its two largest axes)
    axes = np.argsort(wall_size)  # small -> large
    a1, a2 = axes[-2], axes[-1]
    open_size = np.zeros(3)
    open_size[a1] = wall_size[a1] * params.rel_width
    open_size[a2] = wall_size[a2] * params.rel_height
    # Thin thickness along the smallest axis (wall normal direction)
    aN = axes[0]
    open_size[aN] = max(0.02, wall_size[aN] * 0.5)  # cutter thickness

    # Build cutter cube centered at wall center (so opening is central)
    cutter = bproc.object.create_primitive("cube", scale=list(open_size * 0.5))
    cutter.set_location(wall_center)
    cutter.set_name("WindowCutter")
    if room_id is not None:
        cutter.set_cp("room_id", room_id)

    # Try boolean difference
    applied = False
    if params.cut_method == "boolean":
        try:
            bo = wall.get_blender_obj()
            co = cutter.get_blender_obj()
            mod = bo.modifiers.new(name="CutWindow", type='BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.solver = 'FAST'
            mod.object = co
            bpy.context.view_layer.objects.active = bo
            bpy.ops.object.modifier_apply(modifier=mod.name)
            applied = True
        except Exception:
            applied = False

    if not applied:
        # Overlay fallback: push the overlay slightly towards camera side
        # (we can't know the normal easily; offset along smallest bbox axis)
        off = np.zeros(3)
        off[aN] = 0.02
        cutter.set_location(wall_center + off)

    # Create a frame around the opening
    frame_segments = []
    frame_mat = make_opaque_material("window_frame", color=(0.1, 0.1, 0.1), roughness=0.6)

    # Two extents for rectangle in a1,a2 plane; thickness along aN
    fw = open_size[a1]
    fh = open_size[a2]
    t = params.frame_thickness

    # Helper to make a thin rectangular bar centered at given offset in a1/a2 plane
    def _bar(offset_a1, offset_a2, size_a1, size_a2):
        size = np.zeros(3)
        size[a1] = size_a1
        size[a2] = size_a2
        size[aN] = open_size[aN]
        bar = bproc.object.create_primitive("cube", scale=list(size * 0.5))
        loc = wall_center.copy()
        loc[a1] += offset_a1
        loc[a2] += offset_a2
        bar.set_location(loc)
        for i in range(len(bar.get_materials())):
            bar.set_material(i, frame_mat)
        bar.set_name("WindowFrameBar")
        bar.set_cp("inst_mark", "window_frame")
        if room_id is not None:
            bar.set_cp("room_id", room_id)
        return bar

    # Top, bottom, left, right bars
    frame_segments.append(_bar(0.0, +(fh/2 + t/2), fw + 2*t, t))
    frame_segments.append(_bar(0.0, -(fh/2 + t/2), fw + 2*t, t))
    frame_segments.append(_bar(+(fw/2 + t/2), 0.0, t, fh))
    frame_segments.append(_bar(-(fw/2 + t/2), 0.0, t, fh))

    # Glass pane at center
    pane_size = np.zeros(3)
    pane_size[a1] = fw
    pane_size[a2] = fh
    pane_size[aN] = params.glass_thickness

    pane = bproc.object.create_primitive("cube", scale=list(pane_size * 0.5))
    pane.set_location(wall_center)
    glass_mat = make_glass_material("window_glass", params.ior, params.roughness, params.color)
    for i in range(len(pane.get_materials())):
        pane.set_material(i, glass_mat)
    pane.set_name("WindowGlassPane")
    pane.set_cp("inst_mark", "glass_window_pane")
    if room_id is not None:
        pane.set_cp("room_id", room_id)

    # Remove cutter
    try:
        cutter.delete()
    except Exception:
        pass

    return wall, pane, None

# --- Camera interest boosting ----------------------------------------------

def boost_interest_scores_for_transparency(interest_score_setting: dict) -> dict:
    """Raise weights so cameras prefer seeing windows/containers.
    Modifies the dict in-place and returns it.
    """
    # Front3D categories
    interest_score_setting["window"] = max(interest_score_setting.get("window", 0), 10)
    # Common furniture to attract views as context
    for k in ["chair", "sofa", "table", "bed"]:
        interest_score_setting[k] = max(interest_score_setting.get(k, 0), 10)
    return interest_score_setting

import csv

def get_last_label_id_from_csv(csv_path):
    last_id = None
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 보통 [raw_id,name,mapped_id] 형태가 많음. mapped_id(또는 마지막 컬럼)를 사용.
        for row in reader:
            if not row:
                continue
            try:
                cand = int(row[-1])
                last_id = cand if (last_id is None or cand > last_id) else last_id
            except Exception:
                continue
    if last_id is None:
        last_id = 0
    return last_id


def create_glass_container_with_toy(
        mapping_file,
        assets_dir,
        in_room_objects,
        floor_min,
        floor_max,
        make_glass_material,
        make_principled_mat):
    
    # 먼저 토이를 로드해서 실제 크기를 측정
    candidate_files = []
    candidate_files += list(assets_dir.glob("**/*.blend"))
    candidate_files += list(assets_dir.glob("**/*.obj"))
    candidate_files += list(assets_dir.glob("**/*.ply"))
    if len(candidate_files) == 0:
        raise Exception(f"No asset files found in {assets_dir}")
    toy_path = str(random.choice(candidate_files))
    if toy_path.lower().endswith(".blend"):
        toy_entities = load_blend(toy_path, obj_types=['mesh'])
        toy_parts = [e for e in toy_entities if hasattr(e, 'add_material') and hasattr(e, 'get_bound_box')]
    else:
        toy_parts = load_obj(toy_path)

    # 토이의 실제 bbox 계산 (스케일 적용 전 원본 크기)
    toy_mins = []
    toy_maxs = []
    for part in toy_parts:
        bb = part.get_bound_box()
        toy_mins.append(np.min(bb, axis=0))
        toy_maxs.append(np.max(bb, axis=0))
    toy_bb_min = np.min(np.stack(toy_mins, axis=0), axis=0)
    toy_bb_max = np.max(np.stack(toy_maxs, axis=0), axis=0)
    toy_original_size = toy_bb_max - toy_bb_min
    toy_original_size[toy_original_size <= 1e-6] = 1e-6

    # 토이의 기하학적 중심을 계산합니다.
    toy_center_offset = (toy_bb_min + toy_bb_max) / 2.0

    # 모든 파트의 위치를 오프셋만큼 이동시켜 전체 토이의 중심을 원점으로 맞춥니다.
    for part in toy_parts:
        part.set_location(part.get_location() - toy_center_offset)

    # 컨테이너 기본 치수 설정
    frame_t = 0.03                # 프레임 두께 (눈에 보이도록 굵게)
    glass_t = 0.003               # 유리 두께

    # 토이가 들어갈 수 있는 최소 내부 공간 계산 (여유 마진 포함)
    inner_margin = 0.1  # 토이와 컨테이너 벽 사이 최소 간격
    min_inner_size = toy_original_size + 2 * inner_margin

    # 컨테이너 외벽 두께 고려
    wall_thickness = max(frame_t, glass_t) * 2
    min_container_size = min_inner_size + wall_thickness

    # 랜덤 스케일 팩터를 적용하되, 최소 크기는 보장
    x_scale_factor = np.random.uniform(1.0, 2.0)
    y_scale_factor = np.random.uniform(1.0, 2.0)
    z_scale_factor = np.random.uniform(1.0, 2.0)

    # 기본 크기에 스케일 적용하되, 최소 크기보다 작아지지 않도록 보장
    base_size = np.array([0.6, 0.45, 0.6])
    scaled_size = base_size * np.array([x_scale_factor, y_scale_factor, z_scale_factor])
    final_size = np.maximum(scaled_size, min_container_size)

    sx, sy, sz = final_size[0], final_size[1], final_size[2]

    glass_mat = make_glass_material()
    frame_mat = make_principled_mat("FrameBlack", base_color=(0, 0, 0, 1), roughness=0.25, metallic=0.0)
    toy_mat = make_principled_mat("ToyMat", base_color=(0.9, 0.2, 0.2, 1.0), roughness=0.4, metallic=0.0)

    # 물리용 프록시(렌더 숨김). 실제 렌더는 패널/프레임이 담당
    proxy = bproc.object.create_primitive("CUBE")
    proxy.set_name("glass_container_proxy")
    proxy.set_scale([sx/2, sy/2, sz/2])
    proxy.hide(True)

    # 유리 패널 6면 (얇은 큐브)
    panes = []
    # +X / -X
    for dir_x in [1, -1]:
        p = bproc.object.create_primitive("CUBE")
        p.set_scale([glass_t/2, sy/2, sz/2])
        p.set_location([dir_x * sx/2, 0, 0])
        p.set_parent(proxy)
        p.add_material(glass_mat)
        panes.append(p)
    # +Y / -Y
    for dir_y in [1, -1]:
        p = bproc.object.create_primitive("CUBE")
        p.set_scale([sx/2, glass_t/2, sz/2])
        p.set_location([0, dir_y * sy/2, 0])
        p.set_parent(proxy)
        p.add_material(glass_mat)
        panes.append(p)
    # +Z / -Z (뚜껑 포함)
    for dir_z in [1, -1]:
        p = bproc.object.create_primitive("CUBE")
        p.set_scale([sx/2, sy/2, glass_t/2])
        p.set_location([0, 0, dir_z * sz/2])
        p.set_parent(proxy)
        p.add_material(glass_mat)
        panes.append(p)

    transparent_cat_id = get_last_label_id_from_csv(mapping_file)

    # 검정 프레임 12개 빔 (코너 기둥 4 + 상단/하단 링 8)
    frames = []
    # 코너 수직 기둥 4
    for dx in [1, -1]:
        for dy in [1, -1]:
            f = bproc.object.create_primitive("CUBE")
            f.set_scale([frame_t/2, frame_t/2, sz/2])
            f.set_location([dx * sx/2, dy * sy/2, 0])
            f.set_parent(proxy)
            f.add_material(frame_mat)
            frames.append(f)
    # 상단/하단 X 빔 (y=±, z=±)
    for dy in [1, -1]:
        for dz in [1, -1]:
            f = bproc.object.create_primitive("CUBE")
            f.set_scale([sx/2, frame_t/2, frame_t/2])
            f.set_location([0, dy * sy/2, dz * sz/2])
            f.set_parent(proxy)
            f.add_material(frame_mat)
            frames.append(f)
    # 상단/하단 Y 빔 (x=±, z=±)
    for dx in [1, -1]:
        for dz in [1, -1]:
            f = bproc.object.create_primitive("CUBE")
            f.set_scale([frame_t/2, sy/2, frame_t/2])
            f.set_location([dx * sx/2, 0, dz * sz/2])
            f.set_parent(proxy)
            f.add_material(frame_mat)
            frames.append(f)

    # 각 패널에 inst_mark 부여
    for i, p in enumerate(panes):
        p.set_cp("inst_mark", f"glass_pane_{i}")
        p.set_cp("category_id", int(transparent_cat_id)) 
    for i, p in enumerate(frames):
        p.set_cp("inst_mark", f"glass_frame_{i}")
        p.set_cp("category_id", int(transparent_cat_id))

    # 컨테이너 전체 전역 스케일을 랜덤으로 변조 (자식들 포함 스케일됨)
    # 방 바닥 크기에 맞춰 상한 제한, 하한은 더 크게 설정하여 존재감을 높임
    floor_w = float(floor_max[0] - floor_min[0])
    floor_h = float(floor_max[1] - floor_min[1])
    place_margin = 0.30
    desired_min_scale, desired_max_scale = 1.5, 2.0
    max_scale_x = max(0.5, (floor_w - 2 * place_margin) / sx)
    max_scale_y = max(0.5, (floor_h - 2 * place_margin) / sy)
    allowed_max_scale = max(0.5, min(max_scale_x, max_scale_y, desired_max_scale))

    # 천장 높이에 따른 전역 스케일 상한 제한(수직 여유 확보)
    czmin_room = None
    for _o in in_room_objects:
        _nm = _o.get_name().lower()
        if _nm.startswith("ceiling"):
            _bb = _o.get_bound_box()
            _cz = float(np.min(_bb, axis=0)[2])
            czmin_room = _cz if czmin_room is None else min(czmin_room, _cz)
    if czmin_room is not None:
        room_height = float(czmin_room - floor_max[2])
        if room_height > 0:
            # 컨테이너 전체 높이(eff_sz)가 천장과 바닥 사이에 들어가도록 제한
            z_clearance = 0.06  # 소량 여유
            max_scale_z = max(0.5, (room_height - z_clearance) / sz)
            allowed_max_scale = max(0.5, min(allowed_max_scale, max_scale_z))

    if allowed_max_scale < desired_min_scale:
        global_scale = max(0.9, allowed_max_scale * 0.95)
    else:
        global_scale = float(np.random.uniform(desired_min_scale, allowed_max_scale))
    proxy.set_scale([sx/2 * global_scale, sy/2 * global_scale, sz/2 * global_scale])

    # 토이를 컨테이너 내부에 맞게 배치 및 스케일링
    toy_base = Path(toy_path).stem
    for idx, part in enumerate(toy_parts):
        part.set_parent(proxy)
        part.set_location([0, 0, 0])
        # 원본 재질/색을 그대로 둠 (재질이 없더라도 임의 재질 추가하지 않음)
        try:
            _ = part.get_materials()
        except Exception:
            pass
        try:
            part.set_cp("inst_mark", f"toy_{toy_base}_{idx}")
        except Exception:
            pass
        try:
            part.set_cp("category_id", 0)
        except Exception:
            pass

    # 컨테이너 내부 유효 공간(전역 스케일 반영) 계산
    eff_sx, eff_sy, eff_sz = sx * global_scale, sy * global_scale, sz * global_scale
    # 두께도 전역 스케일에 의해 커지므로 이를 반영
    # after computing `inner`
    gap_min = 0.06  # 기존 0.02 → 0.06
    inner = np.array([eff_sx, eff_sy, eff_sz]) - 2*np.array([max(frame_t, glass_t) * global_scale + gap_min]*3)

    inner = np.maximum(inner, 1e-3)

    # 토이 스케일 정규화: 컨테이너 내부 공간에 맞게 스케일링
    # 토이의 현재 크기는 toy_original_size이므로, 이를 내부 공간에 맞춤
    safety_factor = 0.8  # 여유 공간 확보
    scale_factor = float(np.min(inner / toy_original_size) * safety_factor)
    for part in toy_parts:
        part.set_scale([scale_factor, scale_factor, scale_factor])
        part.set_location([0, 0, 0])

    return proxy, panes, frames, toy_parts, eff_sx, eff_sy, eff_sz, place_margin