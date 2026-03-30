import blenderproc as bproc
import argparse
import os
import numpy as np
import random
from pathlib import Path
import json


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
		wm = None
		if hasattr(child, "blender_obj") and child.blender_obj is not None:
			wm = child.blender_obj.matrix_world.copy()
		child.set_parent(parent)
		if wm is not None:
			child.blender_obj.matrix_world = wm
		return True
	except Exception:
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

# --- Main: create glass container around a duck model (no boolean ops) ---
def create_glass_container_with_duck(location, duck_model_path):
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

    # 2) World-space bbox of all loaded duck sub-objects to determine size
    all_corners = []
    for obj in loaded_duck_objects:
        local_corners = [tuple(v[:]) for v in obj.blender_obj.bound_box]
        local2world = obj.get_local2world_mat()
        for c in local_corners:
            world_c = (local2world @ np.array([c[0], c[1], c[2], 1.0]))[:3]
            all_corners.append(world_c)
    
    if not all_corners:
        duck_size = np.array([1.0, 1.0, 1.0])
    else:
        all_corners = np.asarray(all_corners)
        min_corner = all_corners.min(axis=0)
        max_corner = all_corners.max(axis=0)
        duck_size = (max_corner - min_corner)

    # 3) Create the main glass container at the target location
    padding = 0.5  # meters of clearance around duck
    container_dims = duck_size + padding  # (X,Y,Z) external dimensions
    # Slightly increase container to avoid z-fighting and give frame room
    container_dims = container_dims * 1.05

    outer_scale = (container_dims / 2.0).tolist()
    glass_container = bproc.object.create_primitive('CUBE', location=location, scale=outer_scale)
    glass_container.set_name("GlassContainer")

    # 4) Move duck objects to the container's location and parent them
    if all_corners.size > 0:
        # Recalculate duck center based on its own bounding box in world space
        # This ensures correct centering regardless of the duck's internal origin
        min_corner_world = np.min(all_corners, axis=0)
        max_corner_world = np.max(all_corners, axis=0)
        duck_world_center = (min_corner_world + max_corner_world) / 2.0
        
        # Calculate offset to move the duck's center to the container's location
        offset = location - duck_world_center
        for obj in loaded_duck_objects:
            obj.set_location(obj.get_location() + offset)
    
    for obj in loaded_duck_objects:
        obj.set_parent(glass_container, keep_transform=True)

    # 5) Configure glass container (solidify, materials)
    bo = glass_container.blender_obj
    solid = bo.modifiers.new(name="Solidify", type='SOLIDIFY')
    solid.thickness = 0.05
    solid.offset = 1.0
    solid.use_rim = True
    solid.material_offset = 0

    # Materials (robust to Blender 3.x / 4.x)
    glass_mat = bproc.material.create("ContainerGlass")
    glass_mat.set_principled_shader_value("Base Color", [0.98, 0.98, 1.0, 1.0])
    glass_mat.set_principled_shader_value("Roughness", 0.05) # Slightly increase roughness for more realistic reflections
    glass_mat.set_principled_shader_value("IOR", 1.52)
    glass_mat.set_principled_shader_value("Alpha", 1.0)

    bsdf = next((n for n in glass_mat.blender_obj.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf:
        if "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 1.0
        elif "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
    
    assign_material(glass_container, glass_mat, 0)

    # 6) Build and parent frame edges
    frame_mat = bproc.material.create("BlackFrame")
    frame_mat.set_principled_shader_value("Base Color", [0.05, 0.05, 0.05, 1.0])
    frame_mat.set_principled_shader_value("Metallic", 0.9)
    frame_mat.set_principled_shader_value("Roughness", 0.1)

    half_x, half_y, half_z = container_dims[0]/2.0, container_dims[1]/2.0, container_dims[2]/2.0
    edge_thickness = 0.2
    # remove outward push; place edges right at the shell border (inside/outside tolerance handled by thickness)
    #    frame_out_offset = glass_thickness / 2.0
    # place edges close to container faces: offset = half - half_edge_thickness
    edge_offset_x = max(0.0, half_x - edge_thickness/2.0)
    edge_offset_y = max(0.0, half_y - edge_thickness/2.0)
    edge_offset_z = max(0.0, half_z - edge_thickness/2.0)

    frame_edges = []
    def add_edge(world_pos, scale):
        e = bproc.object.create_primitive('CUBE', location=world_pos, scale=scale)
        assign_material(e, frame_mat, 0)
        # parent to container and keep transform so it follows the container
        try:
            e.set_parent(glass_container, keep_transform=True)
        except Exception:
            # fallback if API differs
            pass
        frame_edges.append(e)

    # X-direction edges (long along X)
    for ys in (-1, 1):
        for zs in (-1, 1):
            pos = [location[0], location[1] + ys * edge_offset_y, location[2] + zs * edge_offset_z]
            scale = [half_x, edge_thickness/2.0, edge_thickness/2.0]
            add_edge(pos, scale)

    # Y-direction edges (long along Y)
    for xs in (-1, 1):
        for zs in (-1, 1):
            pos = [location[0] + xs * edge_offset_x, location[1], location[2] + zs * edge_offset_z]
            scale = [edge_thickness/2.0, half_y, edge_thickness/2.0]
            add_edge(pos, scale)

    # Z-direction edges (long along Z)
    for xs in (-1, 1):
        for ys in (-1, 1):
            pos = [location[0] + xs * edge_offset_x, location[1] + ys * edge_offset_y, location[2]]
            scale = [edge_thickness/2.0, edge_thickness/2.0, half_z]
            add_edge(pos, scale)

    return [glass_container] + loaded_duck_objects + frame_edges


def find_bed_surface(loaded_objects):
    """Find bed objects and return a suitable position above the bed surface."""
    # Look for bed objects
    beds = bproc.filter.by_attr(loaded_objects, "name", ".*[Bb]ed.*", regex=True)
    
    if not beds:
        print("No bed objects found, using scene center as fallback")
        return find_scene_center_and_surface(loaded_objects)
    
    print(f"Found {len(beds)} bed object(s)")
    
    # Get the first bed's center and top surface
    bed = beds[0]
    
    # Calculate bed's bounding box in world coordinates
    local2world = bed.get_local2world_mat()
    world_corners = []
    for corner in bed.get_bound_box():
        world_corner = (local2world @ np.array([corner[0], corner[1], corner[2], 1.0]))[:3]
        world_corners.append(world_corner)
    world_corners = np.asarray(world_corners)
    
    min_corner = world_corners.min(axis=0)
    max_corner = world_corners.max(axis=0)
    
    # Center of the bed (X, Y) and position above the top surface (Z)
    center_x = (min_corner[0] + max_corner[0]) / 2.0
    center_y = (min_corner[1] + max_corner[1]) / 2.0
    surface_z = max_corner[2] + 0.1  # 10cm above bed surface
    
    container_location = [center_x, center_y, surface_z]
    
    print(f"Placing container at: {container_location}")
    return container_location


def find_scene_center_and_surface(loaded_objects):
    """Find the center of the scene and a suitable surface for placing objects."""
    # Get all floor objects
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    
    if not floors:
        # Fallback to scene bounding box center
        all_locations = [obj.get_location() for obj in loaded_objects if hasattr(obj, 'get_location')]
        if all_locations:
            center = np.mean(all_locations, axis=0)
            center[2] += 0.1 # Place slightly above the average height
            return center.tolist()
        else:
            return [0, 0, 0.1]
    
    # Calculate floor center and height
    floor_locations = []
    max_z = -float('inf')
    
    for floor in floors:
        loc = floor.get_location()
        floor_locations.append(loc)
        # Get the highest point of floors
        bbox = floor.get_bound_box()
        floor_max_z = max([loc[2] + point[2] for point in bbox])
        max_z = max(max_z, floor_max_z)
    
    if floor_locations:
        center_xy = np.mean(floor_locations, axis=0)[:2]
        surface_height = max_z + 0.05  # Slightly above the floor
        return [center_xy[0], center_xy[1], surface_height]
    else:
        return [0, 0, 0.1]


if __name__ == '__main__':
    '''Parse folders / file paths'''
    args = parse_args()
    front_folder, future_folder, front_3D_texture_folder, cc_material_folder = get_folders(args)
    front_json = front_folder.joinpath(args.front_json)

    if not front_folder.exists() or not future_folder.exists() \
            or not front_3D_texture_folder.exists() or not cc_material_folder.exists():
        raise Exception("One of these folders does not exist!")

    scene_name = front_json.name[:-len(front_json.suffix)]
    print('Loading scene: %s.' % (scene_name))

    # Initialize BlenderProc with error handling
    try:
        print("Initializing BlenderProc...")
        bproc.init()
    except RuntimeError as e:
        if "IDPropertyGroup changed size during iteration" in str(e):
            print("Warning: BlenderProc initialization issue, trying alternative approach...")
            # Alternative initialization approach
            import bpy
            # Clear the scene manually first
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete(use_global=False)
            # Now try init again
            bproc.init()
        else:
            raise e
    
    # Set up renderer (try multiple approaches for compatibility)
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

    # Load mapping file (optional)
    mapping = None
    try:
        mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
        mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
        print("Successfully loaded label mapping")
    except Exception as e:
        print(f"Warning: Could not load label mapping: {e}")
        print("Continuing without label mapping...")

    # Set light bounces for high quality glass rendering
    try:
        bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                         transmission_bounces=200, transparent_max_bounces=200)
    except:
        print("Warning: Could not set light bounces")
    
    # Set camera intrinsic parameters
    bproc.camera.set_intrinsics_from_blender_params(lens=args.fov / 180 * np.pi, image_width=args.res_x,
                                                    image_height=args.res_y,
                                                    lens_unit="FOV")

    # Read 3D Future model info
    with open(future_folder.joinpath('model_info_revised.json'), 'r') as f:
        model_info_data = json.load(f)
    model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for
                         m in model_info_data}

    # Load the Front3D objects
    print("Loading Front3D scene...")
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

    # -------------------------------------------------------------------------
    #          Apply improved materials
    # -------------------------------------------------------------------------
    print("Loading and applying CC materials...")
    cc_materials = bproc.loader.load_ccmaterials(args.cc_material_folder, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])

    # Apply materials to floors
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
    for floor in floors:
        for i in range(len(floor.get_materials())):
            floor.set_material(i, random.choice(cc_materials))

    # Apply wood materials to baseboards and doors
    baseboards_and_doors = bproc.filter.by_attr(loaded_objects, "name", "Baseboard.*|Door.*", regex=True)
    wood_floor_materials = bproc.filter.by_cp(cc_materials, "asset_name", "WoodFloor.*", regex=True)
    for obj in baseboards_and_doors:
        for i in range(len(obj.get_materials())):
            obj.set_material(i, random.choice(wood_floor_materials))

    # Apply marble materials to walls
    walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
    marble_materials = bproc.filter.by_cp(cc_materials, "asset_name", "Marble.*", regex=True)
    for wall in walls:
        for i in range(len(wall.get_materials())):
            wall.set_material(i, random.choice(marble_materials))

    # -------------------------------------------------------------------------
    #          Add Glass Container with Duck on Bed
    # -------------------------------------------------------------------------
    print("Finding bed and adding glass container with duck...")
    duck_model_path = "/home/user/ssd/datasets/Blender/free-rubber-duck-3d-model/source/duck.blend"
    
    # Find the center of the scene to place the container
    container_location = find_bed_surface(loaded_objects)
    
    # Create glass container with duck using exact function from main_debug.py
    all_container_objects = create_glass_container_with_duck(container_location, duck_model_path)
    
    # Enable physics for bed surfaces to act as collision surfaces
    beds = bproc.filter.by_attr(loaded_objects, "name", ".*[Bb]ed.*", regex=True)
    for bed in beds:
        bed.enable_rigidbody(False)  # Static collision object
        
    # Enable physics for floors as well
    floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True) 
    for floor in floors:
        floor.enable_rigidbody(False)  # Static collision object
    
    # Enable physics for the glass container to make it fall naturally
    if all_container_objects:
        # Find the main container object
        main_container = next((obj for obj in all_container_objects if obj.get_name() == "GlassContainer"), None)
        if main_container:
            main_container.enable_rigidbody(True, mass=2.0)  # Dynamic object
            # Children are already parented, no need to do it again here.
        else:
            print("Warning: Could not find 'GlassContainer' to enable physics.")
    
    # Run physics simulation to let the container settle on the bed
    print("Running physics simulation to place container on bed...")
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3, 
                                                     max_simulation_time=6,
                                                     check_object_interval=1)
    
    print(f"Glass container with duck placed on bed via physics simulation")
    if all_container_objects:
        print(f"Container contains {len(all_container_objects)} objects")
    
    # Update loaded_objects to include container objects
    if all_container_objects:
        loaded_objects.extend(all_container_objects)

    # -------------------------------------------------------------------------
    #          Organize objects in collections for better management
    # -------------------------------------------------------------------------
    try:
        import bpy
        
        # Create collection for glass container
        container_collection = bpy.data.collections.new("GlassContainer_Collection")
        bpy.context.scene.collection.children.link(container_collection)
        
        # Move container objects to the collection
        if all_container_objects:
            for obj in all_container_objects:
                # Remove from all existing collections
                for coll in list(obj.blender_obj.users_collection):
                    coll.objects.unlink(obj.blender_obj)
                # Add to container collection
                container_collection.objects.link(obj.blender_obj)
        
        print("Glass container objects organized in 'GlassContainer_Collection'")
        
    except Exception as e:
        print(f"Collection creation failed: {e}")

    # -------------------------------------------------------------------------
    #          Final setup
    # -------------------------------------------------------------------------
    print("Scene loading complete!")
    print("=" * 60)
    print("SCENE SUMMARY:")
    print(f"- Total objects loaded: {len(loaded_objects)}")
    print(f"- Container location: {container_location}")
    if all_container_objects:
        print(f"- Glass container contains: {len(all_container_objects)} objects")
    print("- Glass container features:")
    print("  * Transparent glass walls with refraction")
    print("  * Black metallic frame edges")
    print("  * Duck object inside (loaded from blend file)")
    print("  * Physics simulation applied")
    print("=" * 60)
    print("Ready for manual rendering in Blender GUI!")
    print("You can now:")
    print("1. Adjust camera positions manually")
    print("2. Fine-tune lighting if needed")
    print("3. Render using Cycles with Optix acceleration")
    print("4. The glass container should show proper transparency and refraction")
    print("Ready for manual rendering in Blender GUI!")
    print("You can now:")
    print("1. Adjust camera positions manually")
    print("2. Fine-tune lighting if needed")
    print("3. Render using Cycles with Optix acceleration")
    print("4. The glass container should show proper transparency and refraction")
