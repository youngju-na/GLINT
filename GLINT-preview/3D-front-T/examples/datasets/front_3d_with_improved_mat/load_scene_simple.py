import blenderproc as bproc
import sys
import argparse
import os
import numpy as np
import random
from pathlib import Path
import json
from mathutils import Vector
import bpy

def world_bbox_minmax(obj):
    """모디파이어 적용 상태(evaluated)의 월드 바운딩박스 min/max 반환.
    obj는 bproc 객체(wrapper)일 수도 있고 bpy.types.Object일 수도 있음.
    """
    import bpy
    from mathutils import Vector
    # bproc wrapper인 경우 내부 bpy 오브젝트 사용
    if hasattr(obj, "blender_obj"):
        bpy_obj = obj.blender_obj
    else:
        bpy_obj = obj
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = bpy_obj.evaluated_get(depsgraph)
    corners = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("front_folder", help="Path to the 3D front file")
    parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
    parser.add_argument("front_3D_texture_folder", help="Path to the 3D FRONT texture folder.")
    parser.add_argument("front_json", help="Path to a 3D FRONT scene json file, e.g.6a0e73bc-d0c4-4a38-bfb6-e083ce05ebe9.json.")
    parser.add_argument('cc_material_folder', nargs='?', default="resources/cctextures",
                        help="Path to CCTextures folder, see the /scripts for the download script.")
    parser.add_argument("--fov", type=int, default=90, help="Field of view of camera.")
    parser.add_argument("--res_x", type=int, default=480, help="Image width.")
    parser.add_argument("--res_y", type=int, default=360, help="Image height.")
    # parse_args() 안
    parser.add_argument("--glass_thickness", type=float, default=0.02, help="Glass wall thickness in meters.")
    parser.add_argument("--frame_thickness", type=float, default=0.05, help="Black frame thickness in meters (edge sticks).")
    parser.add_argument("--frame_gap", type=float, default=0.0015, help="Tiny gap between glass and frame to avoid z-fighting.")
    parser.add_argument("--frame_metallic", type=float, default=0.9, help="Frame material metallic.")
    parser.add_argument("--frame_roughness", type=float, default=0.1, help="Frame material roughness.")

    return parser.parse_args()


def get_folders(args):
    front_folder = Path(args.front_folder)
    future_folder = Path(args.future_folder)
    front_3D_texture_folder = Path(args.front_3D_texture_folder)
    cc_material_folder = Path(args.cc_material_folder)
    return front_folder, future_folder, front_3D_texture_folder, cc_material_folder


def check_name(name, category_name):
    return True if category_name in name.lower() else False


# --- Helper: safe material assignment (creates slots if needed) ---
def assign_material(obj, material, slot_index=0):
    """Assign `material` to `obj` at `slot_index`, creating slots if needed."""
    import bpy
    bo = obj.blender_obj
    mats = bo.data.materials
    while len(mats) <= slot_index:
        mats.append(None)
    mats[slot_index] = material.blender_obj

# NEW: 안전한 부모 설정 (트랜스폼 보존)
def safe_set_parent(child, parent):
	import bpy
	try:
		# 저장: child의 월드 매트릭스
		wm = None
		if hasattr(child, "blender_obj") and child.blender_obj is not None:
			wm = child.blender_obj.matrix_world.copy()
		# BlenderProc API 사용 (no keep_transform)
		child.set_parent(parent)
		# 복원: 월드 매트릭스 유지
		if wm is not None:
			child.blender_obj.matrix_world = wm
		return True
	except Exception:
		# Fallback: 직접 bpy 오브젝트로 부모 설정
		try:
			if hasattr(child, "blender_obj") and hasattr(parent, "blender_obj"):
				wm = child.blender_obj.matrix_world.copy() if child.blender_obj is not None else None
				child.blender_obj.parent = parent.blender_obj
				child.blender_obj.parent_type = 'OBJECT'
				if wm is not None:
					child.blender_obj.matrix_world = wm
				return True
		except Exception:
			return False

def create_glass_container_with_duck(location, duck_model_path,
                                     glass_thickness=0.02,
                                     frame_thickness=0.5,
                                     frame_gap=0.0015,
                                     frame_metallic=0.9,
                                     frame_roughness=0.1):
    """Loads a duck model and encloses it in a glass container with black edges (no boolean)."""
    import bpy

    # 1) Load duck
    try:
        loaded_duck_objects = bproc.loader.load_blend(duck_model_path)
        if not loaded_duck_objects:
            raise RuntimeError("No objects loaded from the blend file.")
        print(f"Successfully loaded {len(loaded_duck_objects)} object(s) from {duck_model_path}")
    except Exception as e:
        print(f"Error loading duck model: {e}")
        return []

    # 2) World-space bbox of all loaded duck sub-objects
    all_corners = []
    for obj in loaded_duck_objects:
        local_corners = [tuple(v[:]) for v in obj.blender_obj.bound_box]
        local2world = obj.get_local2world_mat()
        for c in local_corners:
            world_c = (local2world @ np.array([c[0], c[1], c[2], 1.0]))[:3]
            all_corners.append(world_c)
    all_corners = np.asarray(all_corners)
    min_corner = all_corners.min(axis=0)
    max_corner = all_corners.max(axis=0)
    duck_center = (min_corner + max_corner) / 2.0
    duck_size   = (max_corner - min_corner)

    # Center duck at target location
    offset = location - duck_center
    for obj in loaded_duck_objects:
        obj.set_location(obj.get_location() + offset)

    # 3) Container size & thickness
    padding = 0.5  # meters of clearance around duck
    container_dims = duck_size + padding  # (X,Y,Z) external dimensions
    # Slightly enlarge container to avoid z-fighting and give frame room
    container_dims = container_dims * 1.05

    container_objects = list(loaded_duck_objects)

    # 4) Create a glass container using SOLIDIFY (no boolean)
    # Blender's default cube is 2m across at scale=1 → scale = dims/2 for desired size
    outer_scale = (container_dims / 2.0).tolist()
    glass_container = bproc.object.create_primitive('CUBE', location=location, scale=outer_scale)
    glass_container.set_name("GlassContainer")

    # Solidify for shell thickness
    bo = glass_container.blender_obj
    solid = bo.modifiers.new(name="Solidify", type='SOLIDIFY')
    solid.thickness = glass_thickness
    solid.offset = -1.0
    solid.use_even_offset = True
    solid.use_rim = True
    solid.material_offset = 0

    # 5) Materials (robust to Blender 3.x / 4.x)
    glass_mat = bproc.material.create("ContainerGlass")
    glass_mat.set_principled_shader_value("Base Color", [0.98, 0.98, 1.0, 1.0])
    glass_mat.set_principled_shader_value("Roughness", 0.01)
    glass_mat.set_principled_shader_value("IOR", 1.52)
    # For Cycles, leave Alpha=1 and use Transmission for transparency
    glass_mat.set_principled_shader_value("Alpha", 1.0)

    # Transmission socket name changed in Blender 4.x
    bsdf = None
    for n in glass_mat.blender_obj.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            bsdf = n
            break
    if bsdf is None:
        raise RuntimeError("Principled BSDF node not found in glass material")

    if "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = 1.0           # Blender 3.x
    elif "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0     # Blender 4.x
    else:
        print("Warning: no Transmission input on Principled BSDF; using Alpha as fallback")
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = 0.0

    # Optional (viewport): enable refraction for Eevee-like preview
    glass_mat.blender_obj.blend_method = 'BLEND'
    glass_mat.blender_obj.use_screen_refraction = True

    # SAFE assignment (create slot 0 if missing)
    assign_material(glass_container, glass_mat, 0)

    # 6) Build thin black frame edges (12)
    frame_mat = bproc.material.create("BlackFrame")
    frame_mat.set_principled_shader_value("Base Color", [0.05, 0.05, 0.05, 1.0])
    frame_mat.set_principled_shader_value("Metallic", frame_metallic)
    frame_mat.set_principled_shader_value("Roughness", frame_roughness)

    import bpy
    bpy.context.view_layer.update()

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = world_bbox_minmax(glass_container)
    half_x = (xmax - xmin) * 0.5
    half_y = (ymax - ymin) * 0.5
    half_z = (zmax - zmin) * 0.5
    center = Vector(((xmax + xmin) * 0.5, (ymax + ymin) * 0.5, (zmax + zmin) * 0.5))


    half_x, half_y, half_z = container_dims[0]/2.0, container_dims[1]/2.0, container_dims[2]/2.0
    edge_thickness = min(frame_thickness, max(1e-4, 2.0*min(half_x, half_y, half_z) - 1e-4))

    edge_thickness = min(frame_thickness, max(1e-4, 2.0*min(half_x, half_y, half_z) - 1e-4))
    edge_offset_x = max(0.0, half_x - edge_thickness*0.5 - frame_gap)
    edge_offset_y = max(0.0, half_y - edge_thickness*0.5 - frame_gap)
    edge_offset_z = max(0.0, half_z - edge_thickness*0.5 - frame_gap)

    def add_stick(size_xyz, offset_local_xyz):
        # create stick and set world transform/scale, copy container rotation
        s = Vector(size_xyz)
        o = Vector(offset_local_xyz)
        stick = bproc.object.create_primitive('CUBE')
        stick.set_scale([s.x, s.y, s.z])
        stick.set_location(list(center + o))
        # copy container rotation to stick (fallback to blender_obj if API differs)
        try:
            rot = glass_container.get_rotation_euler()
            stick.set_rotation_euler(rot)
        except Exception:
            try:
                stick.blender_obj.rotation_euler = glass_container.blender_obj.rotation_euler
            except Exception:
                pass
        assign_material(stick, frame_mat, 0)
        # parent to container to follow transformations
        try:
            safe_set_parent(stick, glass_container)
        except Exception:
            pass
        container_objects.append(stick)
        return stick

    off_x = edge_offset_x
    off_y = edge_offset_y
    off_z = edge_offset_z
    et2 = edge_thickness * 0.5

    # X-axis sticks (along X, positioned at +/- off_x on X? actually these are bars along X: length half_x*2)
    # we use size (half_x, et/2, et/2) in user example: length along X = half_x*2 -> set_scale uses half-length in earlier workflows,
    # here bproc.set_scale takes full scale (half of cube default 1->2), but previous code used scale as half-length; to be consistent we keep same pattern:
    add_stick(Vector((half_x, et2, et2)), Vector((+off_x, 0.0, +off_z)))
    add_stick(Vector((half_x, et2, et2)), Vector((-off_x, 0.0, +off_z)))
    add_stick(Vector((half_x, et2, et2)), Vector((+off_x, 0.0, -off_z)))
    add_stick(Vector((half_x, et2, et2)), Vector((-off_x, 0.0, -off_z)))

    # Y-axis sticks (long along Y)
    add_stick(Vector((et2, half_y, et2)), Vector((0.0, +off_y, +off_z)))
    add_stick(Vector((et2, half_y, et2)), Vector((0.0, -off_y, +off_z)))
    add_stick(Vector((et2, half_y, et2)), Vector((0.0, +off_y, -off_z)))
    add_stick(Vector((et2, half_y, et2)), Vector((0.0, -off_y, -off_z)))

    # Z-axis sticks (vertical columns)
    add_stick(Vector((et2, et2, half_z)), Vector((+off_x, +off_y, 0.0)))
    add_stick(Vector((et2, et2, half_z)), Vector((-off_x, +off_y, 0.0)))
    add_stick(Vector((et2, et2, half_z)), Vector((+off_x, -off_y, 0.0)))
    add_stick(Vector((et2, et2, half_z)), Vector((-off_x, -off_y, 0.0)))
 
    container_objects.append(glass_container)
    return container_objects


def find_room_centers(loaded_objects):
    """Find center positions for each room based on floor objects."""
    try:
        # Get all floor objects
        floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
        
        if not floors:
            print("No floor objects found, using default positions")
            return [[0, 0, 1.0], [2, 2, 1.0]]  # Default fallback
        
        room_centers = []
        for floor in floors:
            try:
                floor_location = floor.get_location()
                floor_bbox = floor.get_bound_box()
                
                # Calculate floor center
                min_corner = [min([p[i] for p in floor_bbox]) for i in range(3)]
                max_corner = [max([p[i] for p in floor_bbox]) for i in range(3)]
                floor_center = [(min_corner[i] + max_corner[i])/2 + floor_location[i] for i in range(3)]
                
                # Place container at reasonable height above floor
                container_pos = [floor_center[0], floor_center[1], floor_center[2] + 0.8]
                room_centers.append(container_pos)
                
                print(f"Found room center at: {container_pos}")
                
            except Exception as e:
                print(f"Error processing floor {floor.get_name()}: {e}")
                continue
        
        if not room_centers:
            print("No valid room centers found, using fallback")
            return [[0, 0, 1.0]]
            
        return room_centers
        
    except Exception as e:
        print(f"Error finding room centers: {e}")
        return [[0, 0, 1.0]]


if __name__ == '__main__':
    try:
        print("Starting BlenderProc scene loading...")
        
        # Parse arguments
        args = parse_args()
        front_folder, future_folder, front_3D_texture_folder, cc_material_folder = get_folders(args)
        front_json = front_folder.joinpath(args.front_json)

        if not front_folder.exists() or not future_folder.exists() \
                or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
            raise Exception("One of these folders does not exist!")

        scene_name = front_json.name[:-len(front_json.suffix)]
        print(f'Loading scene: {scene_name}')

        # Initialize BlenderProc
        print("Initializing BlenderProc...")
        bproc.init()
        
        # Set up renderer
        try:
            bproc.renderer.set_render_devices(use_only_cpu=False)
        except AttributeError:
            try:
                bproc.renderer.set_render_device("GPU")
            except:
                print("Warning: Could not set render device")
        
        try:
            bproc.renderer.set_denoiser("OPTIX")
        except:
            print("Warning: Could not set denoiser")
            
        try:
            bproc.renderer.set_max_amount_of_samples(128)
        except:
            print("Warning: Could not set sample count")

        # Set light bounces
        try:
            bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                             transmission_bounces=200, transparent_max_bounces=200)
        except:
            print("Warning: Could not set light bounces")

        # Load mapping file (optional)
        mapping = None
        try:
            mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
            mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
            print("Successfully loaded label mapping")
        except Exception as e:
            print(f"Warning: Could not load label mapping: {e}")
            print("Continuing without label mapping...")

        # Read 3D Future model info
        print("Reading model info...")
        with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
            model_info_data = json.load(f)
        model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' 
                           for m in model_info_data}

        # Load Front3D scene
        print("Loading Front3D objects...")
        try:
            # Try with model_id_to_label first (newer API)
            loaded_objects = bproc.loader.load_front3d(
                json_path=str(front_json),
                future_model_path=str(future_folder),
                front_3D_texture_path=str(front_3D_texture_folder),
                label_mapping=mapping,
                model_id_to_label=model_id_to_label)
        except TypeError:
            # Fall back to older API without model_id_to_label
            print("Using older API without model_id_to_label...")
            if mapping is not None:
                loaded_objects = bproc.loader.load_front3d(
                    json_path=str(front_json),
                    future_model_path=str(future_folder),
                    front_3D_texture_path=str(front_3D_texture_folder),
                    label_mapping=mapping)
            else:
                # Skip label mapping if it's None
                loaded_objects = bproc.loader.load_front3d(
                    json_path=str(front_json),
                    future_model_path=str(future_folder),
                    front_3D_texture_path=str(front_3D_texture_folder))

        print(f"Loaded {len(loaded_objects)} objects from Front3D scene")

        # Apply materials
        print("Loading and applying materials...")
        try:
            cc_materials = bproc.loader.load_ccmaterials(str(cc_material_folder), 
                                                        ["Bricks", "Wood", "Carpet", "Tile", "Marble"])
            
            # Apply to floors
            floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
            for floor in floors:
                if cc_materials:
                    floor.replace_materials(random.choice(cc_materials))

            # Apply to walls
            walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
            for wall in walls:
                if cc_materials:
                    wall.replace_materials(random.choice(cc_materials))
                    
            print(f"Applied materials to {len(floors)} floors and {len(walls)} walls")
                    
        except Exception as e:
            print(f"Warning: Could not load CC materials: {e}")

        # Add glass containers to each room with physics simulation
        print("Adding glass containers to each room with physics simulation...")
        room_centers = find_room_centers(loaded_objects)
        all_container_objects = []
        
        # Enable physics for existing objects (floors, walls, furniture) as passive
        print("Setting up physics for existing objects...")
        for obj in loaded_objects:
            try:
                if hasattr(obj, 'enable_rigidbody'):
                    # Check object type to determine physics settings
                    obj_name = obj.get_name().lower()
                    if 'floor' in obj_name or 'wall' in obj_name:
                        # Floors and walls are passive (static) rigid bodies
                        obj.enable_rigidbody(active=False, collision_shape='MESH')
                    elif any(furniture in obj_name for furniture in ['chair', 'table', 'sofa', 'bed', 'cabinet', 'desk', 'shelf']):
                        # Furniture objects are also passive to act as obstacles
                        obj.enable_rigidbody(active=False, collision_shape='CONVEX_HULL')
            except Exception as e:
                # Skip objects that can't have physics enabled
                continue
        
        for i, room_center in enumerate(room_centers):
            print(f"Adding container {i+1} with physics at room: {room_center}")
            
            # Place container at a lower height for more stable physics
            drop_height = room_center[2] + 0.5  # Just 0.5 meters above floor center
            drop_position = [room_center[0], room_center[1], drop_height]
            
            duck_path = "/home/user/ssd/datasets/Blender/free-rubber-duck-3d-model/source/duck.blend"

            container_objects = create_glass_container_with_duck(
                drop_position, duck_path,
                glass_thickness=args.glass_thickness,
                frame_thickness=args.frame_thickness,
                frame_gap=args.frame_gap,
                frame_metallic=args.frame_metallic,
                frame_roughness=args.frame_roughness
                )

            if not container_objects:
                continue

            main_container = next((obj for obj in container_objects if obj.get_name() == "GlassContainer"), None)
            other_objects = [obj for obj in container_objects if obj != main_container]

            if not main_container:
                print(f"Warning: Could not find 'GlassContainer' in container set {i+1}")
                continue

            # 이미 부모가 설정되어 있으면 생략, 다르면 keep_transform=True로 유지
            for obj in other_objects:
                try:
                    if obj.blender_obj.parent is not main_container.blender_obj:
                        safe_set_parent(obj, main_container)
                except Exception:
                    safe_set_parent(obj, main_container)

            # Enable physics only for the main glass container
            main_container.enable_rigidbody(active=True, collision_shape='CONVEX_HULL', mass=3.0, friction=0.6)
            
            try:
                import bpy
                collection = bpy.data.collections.new(f"DuckContainer_Collection_{i}")
                bpy.context.scene.collection.children.link(collection)

                for obj in container_objects:
                    for coll in list(obj.blender_obj.users_collection):
                        coll.objects.unlink(obj.blender_obj)
                    collection.objects.link(obj.blender_obj)

                print(f"All objects for container {i+1} organized in '{collection.name}'")
            except Exception as e:
                print(f"Collection creation for container {i+1} failed: {e}")
            
            all_container_objects.extend(container_objects)
            loaded_objects.extend(container_objects)
        
        # Run physics simulation
        print("Running physics simulation to settle containers...")
        try:
            # Run physics simulation for enough frames to let objects settle
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=2.0,        # Simulate for at least 2 seconds
                max_simulation_time=5.0,        # But not more than 5 seconds
                check_object_interval=1.0,      # Check every 1 second if objects stopped moving
                object_stopped_location_threshold=0.01,  # Objects are considered stopped if they move less than 1cm
                object_stopped_rotation_threshold=0.1    # Objects are considered stopped if they rotate less than 0.1 radians
            )
            print("Physics simulation completed successfully!")
            
        except Exception as e:
            print(f"Warning: Physics simulation encountered an issue: {e}")
            print("Containers may not be in optimal positions")
            
        print(f"Added {len(room_centers)} glass containers with ducks using physics")

        # -------------------------------------------------------------------------
        #          Organize objects in collections for better management
        # -------------------------------------------------------------------------
        try:
            import bpy
            
            # Create collection for all glass containers
            container_collection = bpy.data.collections.new("GlassContainers_Collection")
            bpy.context.scene.collection.children.link(container_collection)
            
            # Move all container objects to the collection
            for obj in all_container_objects:
                # Remove from all existing collections
                for coll in list(obj.blender_obj.users_collection):
                    coll.objects.unlink(obj.blender_obj)
                # Add to container collection
                container_collection.objects.link(obj.blender_obj)
            
            print("All glass container objects organized in 'GlassContainers_Collection'")
            
        except Exception as e:
            print(f"Collection creation failed: {e}")
            
        print("Scene loading completed successfully!")
        print("=" * 60)
        print("SCENE SUMMARY:")
        print(f"- Total objects: {len(loaded_objects)}")
        print(f"- Glass containers: {len(room_centers)}")
        print(f"- Total container components: {len(all_container_objects)}")
        print("- Glass container features:")
        print("  * Solidify modifier for realistic glass thickness")
        print("  * Transparent glass walls with refraction")
        print("  * Black metallic frame edges")
        print("  * duck objects inside each container")
        print("  * One container per room")
        print("=" * 60)
        print("Scene is ready for manual rendering in Blender!")
        print("You can now adjust camera, lighting, and render manually.")
        print("Collections created:")
        print("- GlassContainers_Collection: All containers and ducks")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        print("All glass container objects organized in 'GlassContainers_Collection'")
