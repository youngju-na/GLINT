import blenderproc as bproc
import sys
import argparse
import os
import numpy as np
import random
from pathlib import Path
import json
import signal
from contextlib import contextmanager
import blenderproc.python.renderer.RendererUtility as RendererUtility
from time import time
from blenderproc.python.loader.ObjectLoader import load_obj
from blenderproc.python.loader.BlendLoader import load_blend


# Ensure the local examples/datasets/front_3d_with_improved_mat/utils package is importable
# when the script is executed from a different working directory (e.g. via commands.sh).
# This prepends the script directory and its parent to sys.path so `from utils...` works.
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
# also add the examples/datasets/front_3d_with_improved_mat folder itself (parent_dir already script_dir)
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# from utils.glass_utils import get_last_label_id_from_csv

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


# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json", help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="examples/datasets/front_3d_with_improved_mat/renderings_single_strength_100_thin_glass",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=200,
                        help="The number of views to render in each scene.")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=960, help="Image width.")
    parser.add_argument("--res_y", type=int, default=540, help="Image height.")
    # 새 옵션: 중앙 근처에서 XY 교차 무시하고 초기화, 가구 상호작용 허용
    parser.add_argument("--init_center_ignore_xy", action="store_true", default=False,
                        help="Initialize container near room center with slight randomness, ignoring XY intersection checks.")
    parser.add_argument("--allow_on_furniture", action="store_true", default=True,
                        help="Allow placing container on/against furniture by enabling furniture as passive colliders and skipping XY overlap validation.")
    # 물리 시뮬레이션에서 천장 충돌 무시 옵션
    parser.add_argument("--ignore_ceiling_in_physics", action="store_true", default=False,
                        help="Exclude ceilings from physics so the container won't collide with them during drop (still rendered).")
    return parser.parse_args()


class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)
    n_cameras = args.n_views_per_scene

    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')

    cam_intrinsic_path = output_folder.joinpath('cam_K.npy')

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Processing scene name: %s.' % (scene_name))

    '''Pass those failure cases'''
    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            sys.exit(0)

    '''Pass already generated scenes.'''
    scene_output_folder = output_folder.joinpath(scene_name)
    existing_n_renderings = 0

    if scene_output_folder.is_dir():
        existing_n_renderings = len(list(scene_output_folder.iterdir()))
        if existing_n_renderings >= n_cameras:
            print(len(list(scene_output_folder.iterdir())), 'images already exist. Skip rendering.')
            sys.exit(0)

    if args.append_to_existing_output:
        n_cameras = n_cameras - existing_n_renderings


    bproc.init()
    # === Renderer quality tweaks for better glass reflections/transmission ===
    # Adaptive sampling with a reasonable cap on samples
    RendererUtility.set_noise_threshold(0.0005)
    RendererUtility.set_max_amount_of_samples(768)
    # Prefer fast GPU denoiser if available, else fall back to Intel OIDN
    # try:
    #     RendererUtility.set_denoiser("OPTIX")
    # except Exception:
    #     RendererUtility.set_denoiser("INTEL")
    # Try to light the scene via an HDRI if available; fall back to dim gray world
    try:
        from blenderproc.python.loader.HavenEnvironmentLoader import (
            set_world_background_hdr_img,
            get_random_world_background_hdr_img_path_from_haven,
        )
        haven_root = os.environ.get("HAVEN_DIR", str(Path("resources/haven").resolve()))
        if os.path.exists(haven_root):
            hdr_path = get_random_world_background_hdr_img_path_from_haven(haven_root)
            set_world_background_hdr_img(hdr_path)
        else:
            RendererUtility.set_world_background([0.3, 0.3, 0.3], strength=5.0)
    except Exception:
        RendererUtility.set_world_background([0.3, 0.3, 0.3], strength=5.0)

    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

    # set the light bounces
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                        transmission_bounces=200, transparent_max_bounces=200)
    # set intrinsic parameters
    bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                    image_height=args.res_y,
                                                    lens_unit="FOV")

    cam_K = bproc.camera.get_intrinsics_as_K_matrix()

    # write camera intrinsics
    if not cam_intrinsic_path.exists():
        np.save(str(cam_intrinsic_path), cam_K)

    # read 3d future model info
    with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
        model_info_data = json.load(f)
    model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for
                            m in
                            model_info_data}

    # load the front 3D objects
    loaded_objects = bproc.loader.load_front3d(
        json_path=str(front_json),
        future_model_path=str(future_folder),
        front_3D_texture_path=str(front_3D_texture_folder),
        label_mapping=mapping,
        model_id_to_label=model_id_to_label)

    # -------------------------------------------------------------------------
    #          Sample materials
    # -------------------------------------------------------------------------
    
    paintedbricks_materials = None
    marble_materials = None 
    cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble", "PaintedBricks"])

    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    for floor in floors:
        # For each material of the object
        for i in range(len(floor.get_materials())):
            floor.set_material(i, random.choice(cc_materials))

    baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
    wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
    for obj in baseboards_and_doors:
        # For each material of the object
        for i in range(len(obj.get_materials())):
            # Replace the material with a random one
            obj.set_material(i, random.choice(wood_floor_materials))

    walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
    # paintedbricks_materials = bproc.filter.by_cp(cc_materials, "asset_name", "PaintedBricks.*", regex=True)
    for wall in walls:
        # For each material of the object
        for i in range(len(wall.get_materials())):
            wall.set_material(i, random.choice(marble_materials))
            # wall.set_material(i, random.choice(paintedbricks_materials))

    # -------------------------------------------------------------------------
    #          Sample camera extrinsics
    # -------------------------------------------------------------------------
    # Init sampler for sampling locations inside the loaded front3D house
    point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

    # (1) 바닥 면적이 가장 큰 방을 선택
    floor_areas = np.array(point_sampler.get_floor_areas())
    selected_floor_idx = int(np.argmax(floor_areas))
    selected_floor = point_sampler.used_floors[selected_floor_idx]

    # (2) 선택 방의 바닥 XY 바운딩박스와 겹치는 레이아웃(벽/천장/문/창문/걸레받이) 포함 +
    #     바닥 위에 놓인 가구/오브젝트 포함
    floor_bbox = selected_floor.get_bound_box()
    floor_min = np.min(floor_bbox, axis=0)
    floor_max = np.max(floor_bbox, axis=0)
    
    
    def xy_overlaps(obj_bbox, fmin, fmax):
        omin = np.min(obj_bbox, axis=0)
        omax = np.max(obj_bbox, axis=0)
        return not (omax[0] < fmin[0] or omin[0] > fmax[0] or omax[1] < fmin[1] or omin[1] > fmax[1])

    in_room_objects = []
    for o in loaded_objects:
        if isinstance(o, bproc.types.MeshObject):
            if o == selected_floor:
                in_room_objects.append(o)
                continue
            name_l = o.get_name().lower()
            # 벽/걸레받이/문/창문/천장은 바닥 XY 범위와 겹칠 때 포함
            if name_l.startswith(("wall", "baseboard", "door", "window", "ceiling")):
                if xy_overlaps(o.get_bound_box(), floor_min, floor_max):
                    in_room_objects.append(o)
            else:
                # 가구 등은 바닥 위에 있으면 포함
                if selected_floor.position_is_above_object(o.get_location(), check_no_objects_in_between=False):
                    in_room_objects.append(o)

    # 방 외 객체 숨기기
    for o in loaded_objects:
        if isinstance(o, bproc.types.MeshObject) and o not in in_room_objects:
            o.delete()
            
    # 천장 z 최소값
    def _get_ceiling_zmin(in_room_objs):
        zmin = None
        for o in in_room_objs:
            nm = o.get_name().lower()
            if nm.startswith("ceiling"):
                bb = o.get_bound_box()
                cz = float(np.min(bb, axis=0)[2])
                zmin = cz if zmin is None else min(zmin, cz)
        return zmin

    ceil_z = _get_ceiling_zmin(in_room_objects)
    room_center = np.array([0.5*(floor_min[0]+floor_max[0]), 0.5*(floor_min[1]+floor_max[1]), selected_floor.get_location()[2]])
    if ceil_z is None:
        # 천장이 없는 경우, 바닥에서 2.7m 가정
        ceil_z = float(floor_max[2] + 2.7)

    RendererUtility.add_indoor_lights(floor_min, floor_max, ceil_z, room_center, strength_scale=1.5)

    # === 외부 배경 차단용 오클루더(바깥쪽 벽, 천장, 바닥) 생성 ===
    # 방의 바깥쪽에 얇은 큐브들을 배치해 외부 회색 배경이 보이지 않게 합니다.
    occluders = []
    try:
        # PaintedBricks 재질을 재사용 (없으면 간단한 대체 재질 생성)
        if paintedbricks_materials:
            occluder_material = random.choice(paintedbricks_materials)
        elif marble_materials:
            occluder_material = random.choice(marble_materials)
        else:
            occluder_material = bproc.material.create("OccluderPaintedBricks")
            occluder_material.set_principled_shader_value("Base Color", (0.8, 0.8, 0.8, 1.0))
            occluder_material.set_principled_shader_value("Roughness", 0.6)

        occluder_thickness = 0.02  # 두께 (m)
        pad = 0.25  # 방 외곽으로 약간 확장
        x_len = float(floor_max[0] - floor_min[0])
        y_len = float(floor_max[1] - floor_min[1])
        half_x = x_len / 2.0 + pad
        half_y = y_len / 2.0 + pad
        occluder_height = float((ceil_z - floor_min[2]) + 0.5)
        z_center = float(floor_min[2] + occluder_height / 2.0)
        room_center_xy = [0.5 * (floor_min[0] + floor_max[0]), 0.5 * (floor_min[1] + floor_max[1])]

        # --- 1. 좌/우/앞/뒤 벽면 오클루더 생성 ---
        # +X / -X
        for sign in [1, -1]:
            w = bproc.object.create_primitive("CUBE")
            w.set_name(f"occluder_wall_x_{'pos' if sign==1 else 'neg'}")
            w.set_scale([occluder_thickness / 2.0, (y_len + 2.0 * pad) / 2.0, occluder_height / 2.0])
            w.set_location([room_center_xy[0] + sign * half_x, room_center_xy[1], z_center])
            w.add_material(occluder_material)
            w.set_cp("category_id", 0)
            occluders.append(w)

        # +Y / -Y
        for sign in [1, -1]:
            w = bproc.object.create_primitive("CUBE")
            w.set_name(f"occluder_wall_y_{'pos' if sign==1 else 'neg'}")
            w.set_scale([(x_len + 2.0 * pad) / 2.0, occluder_thickness / 2.0, occluder_height / 2.0])
            w.set_location([room_center_xy[0], room_center_xy[1] + sign * half_y, z_center])
            w.add_material(occluder_material)
            w.set_cp("category_id", 0)
            occluders.append(w)

        # --- 2. 천장 오클루더 생성 ---
        ceiling_z_max = ceil_z  # 이전에 계산된 천장 높이를 기본값으로 사용
        top_occluder = bproc.object.create_primitive("CUBE")
        top_occluder.set_name("occluder_ceiling")
        top_occluder.set_scale([(x_len + 2.0 * pad) / 2.0, (y_len + 2.0 * pad) / 2.0, occluder_thickness / 2.0])
        top_occluder.set_location([room_center_xy[0], room_center_xy[1], ceiling_z_max + occluder_thickness / 2.0 + 0.05])
        top_occluder.add_material(occluder_material)
        top_occluder.set_cp("category_id", 0)
        occluders.append(top_occluder)
        
        # --- 3. 바닥 오클루더 생성 ---
        floor_z_min = floor_min[2] # 선택된 주 바닥의 최소 높이를 사용
        bottom_occluder = bproc.object.create_primitive("CUBE")
        bottom_occluder.set_name("occluder_floor")
        bottom_occluder.set_scale([(x_len + 2.0 * pad) / 2.0, (y_len + 2.0 * pad) / 2.0, occluder_thickness / 2.0])
        bottom_occluder.set_location([room_center_xy[0], room_center_xy[1], floor_z_min - occluder_thickness / 2.0 - 0.05])
        bottom_occluder.add_material(occluder_material)
        bottom_occluder.set_cp("category_id", 0)
        occluders.append(bottom_occluder)

        # BVH 및 카메라 장애물 검사에 포함되도록 in_room_objects에 추가
        in_room_objects.extend(occluders)
        
    except Exception as e:
        print(f"Failed to create occluder objects: {e}")

    # ================== 투명 컨테이너 + 중앙 토이 생성 및 물리 배치 ==================
    def make_glass_material(name="Glass", ior=1.52, roughness=0.0001):
        mat = bproc.material.create(name)
        mat.set_principled_shader_value("Transmission", 1.0)
        mat.set_principled_shader_value("Roughness", roughness)
        mat.set_principled_shader_value("IOR", ior)
        mat.set_principled_shader_value("Specular", 0.4)
        return mat

    def make_principled_mat(name, base_color=(1.0, 1.0, 1.0, 1.0), roughness=0.35, metallic=0.0):
        mat = bproc.material.create(name)
        mat.set_principled_shader_value("Base Color", base_color)
        mat.set_principled_shader_value("Roughness", roughness)
        mat.set_principled_shader_value("Metallic", metallic)
        return mat
    
    print("--------- start loading toy and placing container with physics ---------")
    # 먼저 토이를 로드해서 실제 크기를 측정
    assets_dir = Path("/home/user/ssd/Codes/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/assets") # path: examples/datasets/front_3d_with_improved_mat/assets
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

    print(f"Loaded toy from {toy_path}, num parts: {len(toy_parts)}")

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

    # 토이의 기하학적 중심
    toy_center_offset = (toy_bb_min + toy_bb_max) / 2.0

    # 모든 파트의 위치를 오프셋만큼 이동시켜 전체 토이의 중심을 원점으로 설정.
    for part in toy_parts:
        part.set_location(part.get_location() - toy_center_offset)

    # 컨테이너 기본 치수 설정
    frame_t = 0.02                # 프레임 두께 (눈에 보이도록 굵게)
    glass_t = 0.005               # 유리 두께
    
    inner_margin = 0.1  # 토이와 컨테이너 벽 사이 최소 간격
    min_inner_size = toy_original_size + 2 * inner_margin
    
    # 컨테이너 외벽 두께 고려
    wall_thickness = max(frame_t, glass_t) * 2
    min_container_size = min_inner_size + wall_thickness
    
    # 랜덤 스케일 팩터를 적용하되, 최소 크기는 보장
    x_scale_factor = np.random.uniform(1.5, 2.0)
    y_scale_factor = np.random.uniform(0.7, 1.3)
    z_scale_factor = np.random.uniform(1.3, 2.0)
    
    # 기본 크기에 스케일 적용하되, 최소 크기보다 작아지지 않도록 보장
    base_size = np.array([0.6, 0.4, 0.6])
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
    # +Z만(뚜껑 제외)
    for dir_z in [-1, 1]:
        p = bproc.object.create_primitive("CUBE")
        p.set_scale([sx/2, sy/2, glass_t/2])
        p.set_location([0, 0, dir_z * sz/2])
        p.set_parent(proxy)
        p.add_material(glass_mat)
        panes.append(p)


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

    print("--------- glass container and toy loaded, start physics placement ---------")

    transparent_cat_id = get_last_label_id_from_csv(mapping_file)
    
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
            print(f"Failed to get materials for toy part {idx} of {toy_base}")
        try:
            part.set_cp("inst_mark", f"toy_{toy_base}_{idx}")
        except Exception:
            print(f"Failed to set inst_mark for toy part {idx} of {toy_base}")
        try:
            part.set_cp("category_id", 0)
        except Exception:
            print(f"Failed to set category_id for toy part {idx} of {toy_base}")

    print(f"Container size (before global scale): sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}, global_scale={global_scale:.3f}")
    
    # 컨테이너 내부 유효 공간(전역 스케일 반영) 계산
    eff_sx, eff_sy, eff_sz = sx * global_scale, sy * global_scale, sz * global_scale
    clearance = 0.02
    # 두께도 전역 스케일에 의해 커지므로 이를 반영
    inner = np.array([eff_sx, eff_sy, eff_sz]) - 2*np.array([max(frame_t, glass_t) * global_scale + clearance]*3)
    inner = np.maximum(inner, 1e-3)

    # 토이 스케일 정규화: 컨테이너 내부 공간에 맞게 스케일링
    # 토이의 현재 크기는 toy_original_size이므로, 이를 내부 공간에 맞춤
    safety_factor = 0.6  # 여유 공간 확보
    scale_factor = float(np.min(inner / toy_original_size) * safety_factor)
    for part in toy_parts:
        part.set_scale([scale_factor, scale_factor, scale_factor])
        part.set_location([0, 0, 0])

    # 배치 충돌 회피: 가구 등과 XY AABB가 겹치지 않는 빈 위치를 탐색
    def _xy_intersects(a_min2, a_max2, b_min2, b_max2):
        return not (a_max2[0] < b_min2[0] or a_min2[0] > b_max2[0] or a_max2[1] < b_min2[1] or a_min2[1] > b_max2[1])

    blocking_objs = []
    pane_ids = set(id(p) for p in panes)
    frame_ids = set(id(f) for f in frames)
    toy_ids = set(id(t) for t in toy_parts)
    for o in in_room_objects:
        oid = id(o)
        if oid in pane_ids or oid in frame_ids or oid in toy_ids:
            continue
        nm = o.get_name().lower()
        if nm.startswith(("floor", "wall", "ceiling", "door", "window", "baseboard")):
            continue
        blocking_objs.append(o)

    
    print("--------- start placing container with physics ---------")
    
    # 컨테이너를 방 내부 랜덤 위치 상공에 배치 후 물리 시뮬레이션으로 낙하 정착
    # 방의 패시브 콜라이더 지정 (바닥/벽/문/창문/걸레받이)
    for o in in_room_objects:
        nm = o.get_name().lower()
        if nm.startswith("ceiling") and args.ignore_ceiling_in_physics:
            # 렌더링에는 사용하지만, 물리 충돌에서는 제외
            continue
        if nm.startswith("floor") or nm.startswith("wall") or nm.startswith("ceiling") \
            or nm.startswith("door") or nm.startswith("window") or nm.startswith("baseboard"):
            if not o.has_rigidbody_enabled():
                o.enable_rigidbody(active=False, collision_shape='MESH', friction=0.7)
        elif args.allow_on_furniture and o in blocking_objs:
            # 가구도 패시브로 충돌에 참여하도록 설정하여, 위/옆에 배치 가능
            if not o.has_rigidbody_enabled():
                o.enable_rigidbody(active=False, collision_shape='MESH', friction=0.6)

    # 컨테이너 물리 활성
    proxy.enable_rigidbody(active=True, collision_shape='BOX', friction=0.6)

    # 보조: 천장 z 최소값 수집
    def _get_ceiling_zmin():
        zmin = None
        for o in in_room_objects:
            nm = o.get_name().lower()
            if nm.startswith("ceiling"):
                bb = o.get_bound_box()
                cz = float(np.min(bb, axis=0)[2])
                zmin = cz if zmin is None else min(zmin, cz)
        return zmin

    # 보조: 최종 배치 유효성 검사
    def _validate_final_pose(margin_scale=0.5):
        bb = proxy.get_bound_box()
        pmin = np.min(bb, axis=0)
        pmax = np.max(bb, axis=0)
        # 바닥 AABB 내부(여유 마진 포함)
        margin_x = place_margin * margin_scale
        margin_y = place_margin * margin_scale
        if pmin[0] < floor_min[0] + margin_x or pmax[0] > floor_max[0] - margin_x:
            return False
        if pmin[1] < floor_min[1] + margin_y or pmax[1] > floor_max[1] - margin_y:
            return False
        # 바닥 아래로 침투 금지
        if pmin[2] < floor_min[2] - 1e-3:
            return False
        # XY 충돌 체크(가구 등) — 옵션화
        if not args.allow_on_furniture:
            a_min = pmin[:2]
            a_max = pmax[:2]
            for o in blocking_objs:
                bb_o = o.get_bound_box()
                omin = np.min(bb_o, axis=0)[:2]
                omax = np.max(bb_o, axis=0)[:2]
                if _xy_intersects(a_min, a_max, omin, omax):
                    return False
        # 천장 접촉/관통 방지(존재 시)
        czmin = _get_ceiling_zmin()
        if czmin is not None and pmax[2] > czmin - 0.01:
            return False
        return True

    def sample_free_xy(max_tries=200, buffer=0.10, check_blocking=True):
        half = np.array([eff_sx/2 + buffer, eff_sy/2 + buffer])
        for _ in range(max_tries):
            rx = np.random.uniform(floor_min[0] + place_margin + half[0], floor_max[0] - place_margin - half[0])
            ry = np.random.uniform(floor_min[1] + place_margin + half[1], floor_max[1] - place_margin - half[1])
            if not check_blocking:
                return rx, ry
            a_min = np.array([rx, ry]) - half
            a_max = np.array([rx, ry]) + half
            ok = True
            for o in blocking_objs:
                bb = o.get_bound_box()
                omin = np.min(bb, axis=0)[:2]
                omax = np.max(bb, axis=0)[:2]
                if _xy_intersects(a_min, a_max, omin, omax):
                    ok = False
                    break
            if ok:
                return rx, ry
        return None, None
    
    # 빈 공간/중앙에서 시작 위치 샘플링 (옵션)
    def _sample_start_xy():
        if args.init_center_ignore_xy:
            cx = 0.5 * (floor_min[0] + floor_max[0])
            cy = 0.5 * (floor_min[1] + floor_max[1])
            # 방 중앙 주변으로 작은 난수 오프셋
            rad = 0.15 * min(float(floor_max[0]-floor_min[0]), float(floor_max[1]-floor_min[1]))
            rx = cx + np.random.uniform(-rad, rad)
            ry = cy + np.random.uniform(-rad, rad)
            # 바닥 경계와 컨테이너 크기를 고려해 클립
            rx = float(np.clip(rx, floor_min[0] + place_margin + eff_sx/2, floor_max[0] - place_margin - eff_sx/2))
            ry = float(np.clip(ry, floor_min[1] + place_margin + eff_sy/2, floor_max[1] - place_margin - eff_sy/2))
            return rx, ry
        # 기본: 빈 자리 우선 샘플링
        rx, ry = sample_free_xy()
        if rx is None:
            margin = max(place_margin, 0.15)
            rx = np.random.uniform(floor_min[0] + margin + eff_sx/2, floor_max[0] - margin - eff_sx/2)
            ry = np.random.uniform(floor_min[1] + margin + eff_sy/2, floor_max[1] - margin - eff_sy/2)
        return rx, ry

    # 시작 높이
    rz = floor_max[2] + eff_sz + 0.3
    # 천장이 존재하면 시작 높이를 천장 아래로 클램프
    czmin = _get_ceiling_zmin()
    if czmin is not None:
        rz = float(min(rz, czmin - eff_sz/2 - 0.05))

    # 배치-시뮬레이션 재시도 루프
    max_attempts = 50
    placed_ok = False
    for attempt in range(max_attempts):
        rx, ry = _sample_start_xy()
        proxy.set_location([rx, ry, rz])
        yaw = np.random.uniform(0, 2*np.pi)
        proxy.set_rotation_euler([0, 0, yaw])

        # 물리 시뮬레이션 실행 (정착까지)
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2.0, max_simulation_time=8.0)

        if _validate_final_pose():
            placed_ok = True
            break
        else:
            print(f"[Placement] Invalid final pose, retrying... (attempt {attempt+1}/{max_attempts})")
            # change the scale of the container
            

    if not placed_ok:
        print("[Placement] Warning: Failed to place container in a valid pose after retries; proceeding with last pose.")

    # BVH 및 커버리지 계산에 포함될 렌더링 대상 추가(프록시는 숨김 유지)
    in_room_objects.extend(panes)
    in_room_objects.extend(frames)
    in_room_objects.extend(toy_parts)

    # ================== 투명 컨테이너 배치 종료 ==================
    
    
    # Init bvh tree containing all mesh objects (방 내부만)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(in_room_objects)

    # filter some objects from the loaded objects, which are later used in calculating an interesting score
    interest_score_setting = {'ceiling': 0, 'column': 0, 'customizedpersonalizedmodel': 0, 'beam': 0, 'wallinner': 0,
                                'slabside': 0, 'customizedfixedfurniture': 0, 'cabinet/lightband': 0, 'window': 0,
                                'hole': 0, 'customizedplatform': 0, 'baseboard': 0, 'customizedbackgroundmodel': 0,
                                'front': 0, 'walltop': 0, 'wallouter': 0, 'cornice': 0, 'sewerpipe': 0,
                                'smartcustomizedceiling': 0, 'customizedfeaturewall': 0, 'customizedfurniture': 0,
                                'slabtop': 0, 'baywindow': 0, 'door': 0, 'customized_wainscot': 0, 'slabbottom': 0,
                                'back': 0, 'flue': 0, 'extrusioncustomizedceilingmodel': 0,
                                'extrusioncustomizedbackgroundwall': 0, 'floor': 0, 'lightband': 0,
                                'customizedceiling': 0, 'void': 0, 'pocket': 0, 'wallbottom': 0, 'chair': 10, 'sofa': 10,
                                'table': 10, 'bed': 10}
    special_objects = []
    special_object_scores = {}
    for category_name, category_score in interest_score_setting.items():
        special_objects_per_category = [obj.get_cp("category_id") for obj in in_room_objects if check_name(obj.get_name(), category_name)]
        special_objects.extend(special_objects_per_category)
        unique_cat_ids = set(special_objects_per_category)
        for cat_id in unique_cat_ids:
            special_object_scores[cat_id] = category_score

    # --- 안전장치: render_segmap 등에서 필요한 커스텀 프로퍼티가 없을 때 에러 방지 ---
    for o in in_room_objects:
        if not isinstance(o, bproc.types.MeshObject):
            continue
        try:
            _ = o.get_cp("inst_mark")
        except Exception:
            try:
                o.set_cp("inst_mark", "")
            except Exception:
                pass
        try:
            _ = o.get_cp("category_id")
        except Exception:
            try:
                o.set_cp("category_id", 0)
            except Exception:
                pass
