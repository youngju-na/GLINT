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
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.glass_utils import get_last_label_id_from_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json", help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("output_folder", nargs='?', default="examples/datasets/front_3d_with_improved_mat/renderings_map_all_rooms",
                        help="Path to where the data should be saved")
    parser.add_argument("--n_views_per_scene", type=int, default=200,
                        help="(ignored) kept for compatibility")
    parser.add_argument("--n_views_per_room", type=int, default=100,
                        help="Number of camera views to sample per room (floor) across the house")
    parser.add_argument("--append_to_existing_output", type=bool, default=True,
                        help="If append new renderings to the existing ones.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=960, help="Image width.")
    parser.add_argument("--res_y", type=int, default=540, help="Image height.")
    parser.add_argument("--init_center_ignore_xy", action="store_true", default=False,
                        help="Initialize container near room center with slight randomness, ignoring XY intersection checks.")
    parser.add_argument("--allow_on_furniture", action="store_true", default=False,
                        help="Allow placing container on/against furniture by enabling furniture as passive colliders and skipping XY overlap validation.")
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
        output_folder.mkdir(parents=True, exist_ok=True)
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder


def check_name(name, category_name):
    return True if category_name in name.lower() else False


if __name__ == '__main__':
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder, output_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)

    failed_scene_name_file = output_folder.parent.joinpath('failed_scene_names.txt')
    cam_intrinsic_path = output_folder.joinpath('cam_K.npy')

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Processing scene name: %s.' % (scene_name))

    if failed_scene_name_file.is_file():
        with open(failed_scene_name_file, 'r') as file:
            failure_scenes = file.read().splitlines()
        if scene_name in failure_scenes:
            print('File in failure log: %s. Continue.' % (scene_name))
            sys.exit(0)

    scene_output_folder = output_folder.joinpath(scene_name)
    existing_n_renderings = 0
    if scene_output_folder.is_dir():
        existing_n_renderings = len(list(scene_output_folder.iterdir()))
    if args.append_to_existing_output and existing_n_renderings > 0:
        # we will append per-room later, keep behavior simple
        print('Appending to existing output (found %d files).' % existing_n_renderings)

    try:
        with time_limit(900):
            start_time = time()
            bproc.init()
            RendererUtility.set_noise_threshold(0.005)
            RendererUtility.set_max_amount_of_samples(128)

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
                    RendererUtility.set_world_background([0.2, 0.2, 0.2], strength=5.0)
            except Exception:
                RendererUtility.set_world_background([0.2, 0.2, 0.2], strength=5.0)

            mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
            mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

            bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                             transmission_bounces=200, transparent_max_bounces=200)

            bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                            image_height=args.res_y,
                                                            lens_unit="FOV")
            cam_K = bproc.camera.get_intrinsics_as_K_matrix()
            if not cam_intrinsic_path.exists():
                np.save(str(cam_intrinsic_path), cam_K)

            with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
                model_info_data = json.load(f)
            model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for m in model_info_data}

            loaded_objects = bproc.loader.load_front3d(
                json_path=str(front_json),
                future_model_path=str(future_folder),
                front_3D_texture_path=str(front_3D_texture_folder),
                label_mapping=mapping,
                model_id_to_label=model_id_to_label)

            cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble", "PaintedBricks"])

            floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
            for floor in floors:
                for i in range(len(floor.get_materials())):
                    floor.set_material(i, random.choice(cc_materials))

            baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
            wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
            for obj in baseboards_and_doors:
                for i in range(len(obj.get_materials())):
                    obj.set_material(i, random.choice(wood_floor_materials))

            walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
            paintedbricks_materials = bproc.filter.by_cp(cc_materials, "asset_name", "PaintedBricks.*", regex=True)
            for wall in walls:
                for i in range(len(wall.get_materials())):
                    wall.set_material(i, random.choice(paintedbricks_materials))

            # Init sampler and floor info
            point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)
            floor_areas = np.array(point_sampler.get_floor_areas())
            n_floors = len(point_sampler.used_floors)
            # Choose a floor for container placement (largest) but we'll sample cameras in all floors
            selected_floor_idx = int(np.argmax(floor_areas))
            selected_floor = point_sampler.used_floors[selected_floor_idx]

            # Build two lists:
            # - in_room_objects_container: objects considered for container placement / physics (objects overlapping selected_floor)
            # - all_room_meshes: all mesh objects in the house used for BVH and camera obstacle checks
            def xy_overlaps(obj_bbox, fmin, fmax):
                omin = np.min(obj_bbox, axis=0)
                omax = np.max(obj_bbox, axis=0)
                return not (omax[0] < fmin[0] or omin[0] > fmax[0] or omax[1] < fmin[1] or omin[1] > fmax[1])

            in_room_objects_container = []
            all_room_meshes = []
            for o in loaded_objects:
                if isinstance(o, bproc.types.MeshObject):
                    all_room_meshes.append(o)
                    name_l = o.get_name().lower()
                    floor_bbox = selected_floor.get_bound_box()
                    floor_min = np.min(floor_bbox, axis=0)
                    floor_max = np.max(floor_bbox, axis=0)
                    if o == selected_floor:
                        in_room_objects_container.append(o)
                        continue
                    if name_l.startswith(("wall", "baseboard", "door", "window", "ceiling")):
                        if xy_overlaps(o.get_bound_box(), floor_min, floor_max):
                            in_room_objects_container.append(o)
                    else:
                        if selected_floor.position_is_above_object(o.get_location(), check_no_objects_in_between=False):
                            in_room_objects_container.append(o)

            # Do NOT hide other rooms — we sample cameras across the whole house

            def _get_ceiling_zmin(in_room_objs):
                zmin = None
                for o in in_room_objs:
                    nm = o.get_name().lower()
                    if nm.startswith("ceiling"):
                        bb = o.get_bound_box()
                        cz = float(np.min(bb, axis=0)[2])
                        zmin = cz if zmin is None else min(zmin, cz)
                return zmin

            floor_bbox = selected_floor.get_bound_box()
            floor_min = np.min(floor_bbox, axis=0)
            floor_max = np.max(floor_bbox, axis=0)
            ceil_z = _get_ceiling_zmin(in_room_objects_container)
            room_center = np.array([0.5*(floor_min[0]+floor_max[0]), 0.5*(floor_min[1]+floor_max[1]), selected_floor.get_location()[2]])
            if ceil_z is None:
                ceil_z = float(floor_max[2] + 2.7)

            RendererUtility.add_indoor_lights(floor_min, floor_max, ceil_z, room_center, strength_scale=1.0)

            # Create occluders and add to all_room_meshes (so BVH and camera checks include them)
            occluders = []
            try:
                if paintedbricks_materials:
                    occluder_material = random.choice(paintedbricks_materials)
                else:
                    occluder_material = bproc.material.create("OccluderPaintedBricks")
                    occluder_material.set_principled_shader_value("Base Color", (0.8, 0.8, 0.8, 1.0))
                    occluder_material.set_principled_shader_value("Roughness", 0.6)
                occluder_thickness = 0.02
                pad = 0.25
                x_len = float(floor_max[0] - floor_min[0])
                y_len = float(floor_max[1] - floor_min[1])
                half_x = x_len / 2.0 + pad
                half_y = y_len / 2.0 + pad
                occluder_height = float((ceil_z - floor_min[2]) + 0.5)
                z_center = float(floor_min[2] + occluder_height / 2.0)
                room_center_xy = [0.5 * (floor_min[0] + floor_max[0]), 0.5 * (floor_min[1] + floor_max[1])]
                for sign in [1, -1]:
                    w = bproc.object.create_primitive("CUBE")
                    w.set_name(f"occluder_wall_x_{'pos' if sign==1 else 'neg'}")
                    w.set_scale([occluder_thickness / 2.0, (y_len + 2.0 * pad) / 2.0, occluder_height / 2.0])
                    w.set_location([room_center_xy[0] + sign * half_x, room_center_xy[1], z_center])
                    w.add_material(occluder_material)
                    try:
                        w.set_cp("category_id", 0)
                    except Exception:
                        pass
                    occluders.append(w)
                for sign in [1, -1]:
                    w = bproc.object.create_primitive("CUBE")
                    w.set_name(f"occluder_wall_y_{'pos' if sign==1 else 'neg'}")
                    w.set_scale([(x_len + 2.0 * pad) / 2.0, occluder_thickness / 2.0, occluder_height / 2.0])
                    w.set_location([room_center_xy[0], room_center_xy[1] + sign * half_y, z_center])
                    w.add_material(occluder_material)
                    try:
                        w.set_cp("category_id", 0)
                    except Exception:
                        pass
                    occluders.append(w)
                all_room_meshes.extend(occluders)
            except Exception as e:
                print(f"Failed to create occluder walls: {e}")

            # ================== transparent container and toy placement (same as original) ==================
            def make_glass_material(name="Glass", ior=1.52, roughness=0.02):
                mat = bproc.material.create(name)
                mat.set_principled_shader_value("Transmission", 1.0)
                mat.set_principled_shader_value("Roughness", roughness)
                mat.set_principled_shader_value("IOR", ior)
                mat.set_principled_shader_value("Specular", 0.5)
                return mat

            def make_principled_mat(name, base_color=(1.0, 1.0, 1.0, 1.0), roughness=0.35, metallic=0.0):
                mat = bproc.material.create(name)
                mat.set_principled_shader_value("Base Color", base_color)
                mat.set_principled_shader_value("Roughness", roughness)
                mat.set_principled_shader_value("Metallic", metallic)
                return mat

            x_scale_factor = np.random.uniform(1.5, 2.0)
            y_scale_factor = np.random.uniform(1.5, 3.0)
            z_scale_factor = np.random.uniform(1.5, 3.0)
            sx, sy, sz = 0.6 * x_scale_factor, 0.45 * y_scale_factor, 0.45 * z_scale_factor
            frame_t = 0.03
            glass_t = 0.005

            glass_mat = make_glass_material()
            frame_mat = make_principled_mat("FrameBlack", base_color=(0, 0, 0, 1), roughness=0.25, metallic=0.0)
            toy_mat = make_principled_mat("ToyMat", base_color=(0.9, 0.2, 0.2, 1.0), roughness=0.4, metallic=0.0)

            proxy = bproc.object.create_primitive("CUBE")
            proxy.set_name("glass_container_proxy")
            proxy.set_scale([sx/2, sy/2, sz/2])
            proxy.hide(True)

            panes = []
            for dir_x in [1, -1]:
                p = bproc.object.create_primitive("CUBE")
                p.set_scale([glass_t/2, sy/2, sz/2])
                p.set_location([dir_x * sx/2, 0, 0])
                p.set_parent(proxy)
                p.add_material(glass_mat)
                panes.append(p)
            for dir_y in [1, -1]:
                p = bproc.object.create_primitive("CUBE")
                p.set_scale([sx/2, glass_t/2, sz/2])
                p.set_location([0, dir_y * sy/2, 0])
                p.set_parent(proxy)
                p.add_material(glass_mat)
                panes.append(p)
            for dir_z in [1, -1]:
                p = bproc.object.create_primitive("CUBE")
                p.set_scale([sx/2, sy/2, glass_t/2])
                p.set_location([0, 0, dir_z * sz/2])
                p.set_parent(proxy)
                p.add_material(glass_mat)
                panes.append(p)

            transparent_cat_id = get_last_label_id_from_csv(mapping_file)
            for i, p in enumerate(panes):
                p.set_cp("inst_mark", f"glass_pane_{i}")
                p.set_cp("category_id", int(transparent_cat_id))

            frames = []
            for dx in [1, -1]:
                for dy in [1, -1]:
                    f = bproc.object.create_primitive("CUBE")
                    f.set_scale([frame_t/2, frame_t/2, sz/2])
                    f.set_location([dx * sx/2, dy * sy/2, 0])
                    f.set_parent(proxy)
                    f.add_material(frame_mat)
                    frames.append(f)
            for dy in [1, -1]:
                for dz in [1, -1]:
                    f = bproc.object.create_primitive("CUBE")
                    f.set_scale([sx/2, frame_t/2, frame_t/2])
                    f.set_location([0, dy * sy/2, dz * sz/2])
                    f.set_parent(proxy)
                    f.add_material(frame_mat)
                    frames.append(f)
            for dx in [1, -1]:
                for dz in [1, -1]:
                    f = bproc.object.create_primitive("CUBE")
                    f.set_scale([frame_t/2, sy/2, frame_t/2])
                    f.set_location([dx * sx/2, 0, dz * sz/2])
                    f.set_parent(proxy)
                    f.add_material(frame_mat)
                    frames.append(f)

            for i, f in enumerate(frames):
                try:
                    f.set_cp("inst_mark", f"frame_beam_{i}")
                except Exception:
                    pass
                try:
                    f.set_cp("category_id", 0)
                except Exception:
                    pass

            floor_w = float(floor_max[0] - floor_min[0])
            floor_h = float(floor_max[1] - floor_min[1])
            place_margin = 0.30
            desired_min_scale, desired_max_scale = 1.5, 2.0
            max_scale_x = max(0.5, (floor_w - 2 * place_margin) / sx)
            max_scale_y = max(0.5, (floor_h - 2 * place_margin) / sy)
            allowed_max_scale = max(0.5, min(max_scale_x, max_scale_y, desired_max_scale))

            czmin_room = None
            for _o in in_room_objects_container:
                _nm = _o.get_name().lower()
                if _nm.startswith("ceiling"):
                    _bb = _o.get_bound_box()
                    _cz = float(np.min(_bb, axis=0)[2])
                    czmin_room = _cz if czmin_room is None else min(czmin_room, _cz)
            if czmin_room is not None:
                room_height = float(czmin_room - floor_max[2])
                if room_height > 0:
                    z_clearance = 0.06
                    max_scale_z = max(0.5, (room_height - z_clearance) / sz)
                    allowed_max_scale = max(0.5, min(allowed_max_scale, max_scale_z))

            if allowed_max_scale < desired_min_scale:
                global_scale = max(0.9, allowed_max_scale * 0.95)
            else:
                global_scale = float(np.random.uniform(desired_min_scale, allowed_max_scale))
            proxy.set_scale([sx/2 * global_scale, sy/2 * global_scale, sz/2 * global_scale])

            assets_dir = (Path(__file__).resolve().parent / "assets")
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

            toy_base = Path(toy_path).stem
            for idx, part in enumerate(toy_parts):
                part.set_parent(proxy)
                part.set_location([0, 0, 0])
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

            eff_sx, eff_sy, eff_sz = sx * global_scale, sy * global_scale, sz * global_scale
            clearance = 0.02
            inner = np.array([eff_sx, eff_sy, eff_sz]) - 2*np.array([max(frame_t, glass_t) * global_scale + clearance]*3)
            inner = np.maximum(inner, 1e-3)

            mins = []
            maxs = []
            for part in toy_parts:
                bb = part.get_bound_box()
                mins.append(np.min(bb, axis=0))
                maxs.append(np.max(bb, axis=0))
            bb_min = np.min(np.stack(mins, axis=0), axis=0)
            bb_max = np.max(np.stack(maxs, axis=0), axis=0)
            toy_size = bb_max - bb_min
            toy_size[toy_size <= 1e-6] = 1e-6
            scale_factor = float(np.min(inner / toy_size) * 0.8)
            for part in toy_parts:
                part.set_scale([scale_factor, scale_factor, scale_factor])
                part.set_location([0, 0, 0])

            def _xy_intersects(a_min2, a_max2, b_min2, b_max2):
                return not (a_max2[0] < b_min2[0] or a_min2[0] > b_max2[0] or a_max2[1] < b_min2[1] or a_min2[1] > b_max2[1])

            blocking_objs = []
            pane_ids = set(id(p) for p in panes)
            frame_ids = set(id(f) for f in frames)
            toy_ids = set(id(t) for t in toy_parts)
            for o in in_room_objects_container:
                oid = id(o)
                if oid in pane_ids or oid in frame_ids or oid in toy_ids:
                    continue
                nm = o.get_name().lower()
                if nm.startswith(("floor", "wall", "ceiling", "door", "window", "baseboard")):
                    continue
                blocking_objs.append(o)

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

            for o in in_room_objects_container:
                nm = o.get_name().lower()
                if nm.startswith("ceiling") and args.ignore_ceiling_in_physics:
                    continue
                if nm.startswith("floor") or nm.startswith("wall") or nm.startswith("ceiling") \
                   or nm.startswith("door") or nm.startswith("window") or nm.startswith("baseboard"):
                    if not o.has_rigidbody_enabled():
                        o.enable_rigidbody(active=False, collision_shape='MESH', friction=0.7)
                elif args.allow_on_furniture and o in blocking_objs:
                    if not o.has_rigidbody_enabled():
                        o.enable_rigidbody(active=False, collision_shape='MESH', friction=0.6)

            proxy.enable_rigidbody(active=True, collision_shape='BOX', friction=0.6)

            def _get_ceiling_zmin():
                zmin = None
                for o in in_room_objects_container:
                    nm = o.get_name().lower()
                    if nm.startswith("ceiling"):
                        bb = o.get_bound_box()
                        cz = float(np.min(bb, axis=0)[2])
                        zmin = cz if zmin is None else min(zmin, cz)
                return zmin

            def _validate_final_pose(margin_scale=0.5):
                bb = proxy.get_bound_box()
                pmin = np.min(bb, axis=0)
                pmax = np.max(bb, axis=0)
                margin_x = place_margin * margin_scale
                margin_y = place_margin * margin_scale
                if pmin[0] < floor_min[0] + margin_x or pmax[0] > floor_max[0] - margin_x:
                    return False
                if pmin[1] < floor_min[1] + margin_y or pmax[1] > floor_max[1] - margin_y:
                    return False
                if pmin[2] < floor_min[2] - 1e-3:
                    return False
                if not args.allow_on_furniture:
                    a_min = pmin[:2]
                    a_max = pmax[:2]
                    for o in blocking_objs:
                        bb_o = o.get_bound_box()
                        omin = np.min(bb_o, axis=0)[:2]
                        omax = np.max(bb_o, axis=0)[:2]
                        if _xy_intersects(a_min, a_max, omin, omax):
                            return False
                czmin = _get_ceiling_zmin()
                if czmin is not None and pmax[2] > czmin - 0.01:
                    return False
                return True

            def _sample_start_xy():
                if args.init_center_ignore_xy:
                    cx = 0.5 * (floor_min[0] + floor_max[0])
                    cy = 0.5 * (floor_min[1] + floor_max[1])
                    rad = 0.15 * min(float(floor_max[0]-floor_min[0]), float(floor_max[1]-floor_min[1]))
                    rx = cx + np.random.uniform(-rad, rad)
                    ry = cy + np.random.uniform(-rad, rad)
                    rx = float(np.clip(rx, floor_min[0] + place_margin + eff_sx/2, floor_max[0] - place_margin - eff_sx/2))
                    ry = float(np.clip(ry, floor_min[1] + place_margin + eff_sy/2, floor_max[1] - place_margin - eff_sy/2))
                    return rx, ry
                rx, ry = sample_free_xy()
                if rx is None:
                    margin = max(place_margin, 0.15)
                    rx = np.random.uniform(floor_min[0] + margin + eff_sx/2, floor_max[0] - margin - eff_sx/2)
                    ry = np.random.uniform(floor_min[1] + margin + eff_sy/2, floor_max[1] - margin - eff_sy/2)
                return rx, ry

            rz = floor_max[2] + eff_sz + 0.3
            czmin = _get_ceiling_zmin()
            if czmin is not None:
                rz = float(min(rz, czmin - eff_sz/2 - 0.05))

            max_attempts = 5
            placed_ok = False
            for attempt in range(max_attempts):
                rx, ry = _sample_start_xy()
                proxy.set_location([rx, ry, rz])
                yaw = np.random.uniform(0, 2*np.pi)
                proxy.set_rotation_euler([0, 0, yaw])

                bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=2.0, max_simulation_time=8.0)

                if _validate_final_pose():
                    placed_ok = True
                    break
                else:
                    print(f"[Placement] Invalid final pose, retrying... (attempt {attempt+1}/{max_attempts})")

            if not placed_ok:
                print("[Placement] Warning: Failed to place container in a valid pose after retries; proceeding with last pose.")

            # Add placement objects to both lists so BVH and scoring include them
            all_room_meshes.extend(panes)
            all_room_meshes.extend(frames)
            all_room_meshes.extend(toy_parts)
            in_room_objects_container.extend(panes)
            in_room_objects_container.extend(frames)
            in_room_objects_container.extend(toy_parts)

            # Build BVH over all meshes for camera checks
            bvh_tree = bproc.object.create_bvh_tree_multi_objects(all_room_meshes)

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
                special_objects_per_category = [obj.get_cp("category_id") for obj in all_room_meshes if check_name(obj.get_name(), category_name)]
                special_objects.extend(special_objects_per_category)
                unique_cat_ids = set(special_objects_per_category)
                for cat_id in unique_cat_ids:
                    special_object_scores[cat_id] = category_score

            # sample camera poses across all floors
            proximity_checks = {}
            cam_Ts = []
            cam_nums = np.zeros_like(floor_areas, dtype=np.int16)
            cam_nums[:] = args.n_views_per_room
            n_tries = cam_nums * 3

            for floor_id, cam_num_per_scene in enumerate(cam_nums):
                cam2world_matrices = []
                coverage_scores = []
                tries = 0
                while tries < n_tries[floor_id]:
                    height = np.random.uniform(1.4, 1.8)
                    location = point_sampler.sample_by_floor_id(height, floor_id=floor_id)
                    rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
                    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

                    obstacle_check = bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree)
                    coverage_score = bproc.camera.scene_coverage_score(cam2world_matrix, special_objects,
                                                                       special_objects_weight=special_object_scores)
                    if obstacle_check and coverage_score >= 0.5:
                        cam2world_matrices.append(cam2world_matrix)
                        coverage_scores.append(coverage_score)
                        tries += 1
                cam_ids = np.argsort(coverage_scores)[-cam_num_per_scene:]
                for cam_id, cam2world_matrix in enumerate(cam2world_matrices):
                    if cam_id in cam_ids:
                        bproc.camera.add_camera_pose(cam2world_matrix)
                        cam_Ts.append(cam2world_matrix)

            bproc.renderer.enable_normals_output()
            bproc.renderer.enable_depth_output(activate_antialiasing=False)
            data = bproc.renderer.render()
            default_values = {
                "location": [0, 0, 0],
                "cp_inst_mark": '',
                "cp_uid": '',
                "cp_jid": '',
                "cp_room_id": '',
                "category_id": 0,
                "type": ''
            }
            data.update(bproc.renderer.render_segmap(
                map_by=["instance", "class", "cp_uid", "cp_jid", "cp_inst_mark", "cp_room_id", "location", "category_id"],
                default_values=default_values))

            data['cam_Ts'] = cam_Ts
            bproc.writer.write_hdf5(str(scene_output_folder), data,
                                    append_to_existing_output=args.append_to_existing_output)
            print('Time elapsed: %f.' % (time()-start_time))

    except TimeoutException as e:
        print('Time is out: %s.' % scene_name)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)
    except Exception as e:
        print('Failed scene name: %s.' % scene_name)
        print("error:", e)
        with open(failed_scene_name_file, 'a') as file:
            file.write(scene_name + "\n")
        sys.exit(0)
