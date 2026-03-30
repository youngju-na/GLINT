import blenderproc as bproc
import sys
import os
import numpy as np
from mathutils import Vector
import bpy

def world_bbox_minmax(obj):
    if hasattr(obj, "blender_obj"):
        bpy_obj = obj.blender_obj
    else:
        bpy_obj = obj
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = bpy_obj.evaluated_get(depsgraph)
    corners = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))

def assign_material(obj, material, slot_index=0):
    bo = obj.blender_obj
    mats = bo.data.materials
    while len(mats) <= slot_index:
        mats.append(None)
    mats[slot_index] = material.blender_obj

def safe_set_parent(child, parent):
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

def create_glass_container_with_duck(location, duck_model_path,
                                     glass_thickness=0.02,
                                     frame_thickness=0.05,
                                     frame_gap=0.0015,
                                     frame_metallic=0.9,
                                     frame_roughness=0.1,
                                     padding=0.5):
    # Load duck
    loaded = bproc.loader.load_blend(duck_model_path)
    if not loaded:
        raise RuntimeError("Duck blend did not load: " + duck_model_path)

    # compute world bbox of duck group
    all_corners = []
    for o in loaded:
        local_corners = [tuple(v[:]) for v in o.blender_obj.bound_box]
        local2world = o.get_local2world_mat()
        for c in local_corners:
            wc = (local2world @ np.array([c[0], c[1], c[2], 1.0]))[:3]
            all_corners.append(wc)
    all_corners = np.asarray(all_corners)
    min_corner = all_corners.min(axis=0)
    max_corner = all_corners.max(axis=0)
    duck_center = (min_corner + max_corner) / 2.0
    duck_size = (max_corner - min_corner)

    # center duck at requested location
    loc = Vector(location)
    offset = loc - Vector(duck_center)
    for o in loaded:
        o.set_location(o.get_location() + offset)

    # container dims
    container_dims = (duck_size + padding) * 1.05
    container_objects = list(loaded)

    # create outer cube and solidify
    outer_scale = (container_dims / 2.0).tolist()
    glass_container = bproc.object.create_primitive('CUBE', location=list(loc), scale=outer_scale)
    glass_container.set_name("GlassContainer")
    bo = glass_container.blender_obj
    solid = bo.modifiers.new(name="Solidify", type='SOLIDIFY')
    solid.thickness = glass_thickness
    solid.offset = -1.0
    solid.use_even_offset = True
    solid.use_rim = True
    solid.material_offset = 0

    # glass material
    glass_mat = bproc.material.create("ContainerGlass")
    glass_mat.set_principled_shader_value("Base Color", [0.98,0.98,1.0,1.0])
    glass_mat.set_principled_shader_value("Roughness", 0.01)
    glass_mat.set_principled_shader_value("IOR", 1.52)
    glass_mat.set_principled_shader_value("Alpha", 1.0)
    bsdf = None
    for n in glass_mat.blender_obj.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            bsdf = n; break
    if bsdf is not None:
        if "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = 1.0
        elif "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
    glass_mat.blender_obj.blend_method = 'BLEND'
    glass_mat.blender_obj.use_screen_refraction = True
    assign_material(glass_container, glass_mat, 0)

    # frame material
    frame_mat = bproc.material.create("BlackFrame")
    frame_mat.set_principled_shader_value("Base Color", [0.05,0.05,0.05,1.0])
    frame_mat.set_principled_shader_value("Metallic", frame_metallic)
    frame_mat.set_principled_shader_value("Roughness", frame_roughness)

    bpy.context.view_layer.update()
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = world_bbox_minmax(glass_container)
    center = Vector(((xmax + xmin) * 0.5, (ymax + ymin) * 0.5, (zmax + zmin) * 0.5))

    half_x, half_y, half_z = container_dims[0]/2.0, container_dims[1]/2.0, container_dims[2]/2.0
    edge_thickness = min(frame_thickness, max(1e-4, 2.0*min(half_x, half_y, half_z)-1e-4))
    edge_offset_x = max(0.0, half_x - edge_thickness*0.5 - frame_gap)
    edge_offset_y = max(0.0, half_y - edge_thickness*0.5 - frame_gap)
    edge_offset_z = max(0.0, half_z - edge_thickness*0.5 - frame_gap)
    et2 = edge_thickness * 0.5

    def add_stick(size_xyz, offset_local_xyz):
        s = Vector(size_xyz)
        o = Vector(offset_local_xyz)
        stick = bproc.object.create_primitive('CUBE')
        stick.set_scale([s.x, s.y, s.z])
        stick.set_location(list(center + o))
        try:
            rot = glass_container.get_rotation_euler()
            stick.set_rotation_euler(rot)
        except Exception:
            try:
                stick.blender_obj.rotation_euler = glass_container.blender_obj.rotation_euler
            except Exception:
                pass
        assign_material(stick, frame_mat, 0)
        safe_set_parent(stick, glass_container)
        container_objects.append(stick)
        return stick

    off_x, off_y, off_z = edge_offset_x, edge_offset_y, edge_offset_z

    # X bars
    add_stick((half_x, et2, et2), (+off_x, 0.0, +off_z))
    add_stick((half_x, et2, et2), (-off_x, 0.0, +off_z))
    add_stick((half_x, et2, et2), (+off_x, 0.0, -off_z))
    add_stick((half_x, et2, et2), (-off_x, 0.0, -off_z))
    # Y bars
    add_stick((et2, half_y, et2), (0.0, +off_y, +off_z))
    add_stick((et2, half_y, et2), (0.0, -off_y, +off_z))
    add_stick((et2, half_y, et2), (0.0, +off_y, -off_z))
    add_stick((et2, half_y, et2), (0.0, -off_y, -off_z))
    # Z posts
    add_stick((et2, et2, half_z), (+off_x, +off_y, 0.0))
    add_stick((et2, et2, half_z), (-off_x, +off_y, 0.0))
    add_stick((et2, et2, half_z), (+off_x, -off_y, 0.0))
    add_stick((et2, et2, half_z), (-off_x, -off_y, 0.0))

    container_objects.append(glass_container)
    return container_objects

if __name__ == "__main__":
    # 설정: 경로와 위치만 바꿔 사용
    bproc.init()
    duck_blend = "/home/user/ssd/datasets/Blender/free-rubber-duck-3d-model/source/duck.blend"
    location = [0.0, 0.0, 1.0]
    objs = create_glass_container_with_duck(location, duck_blend)
    # 모아서 컬렉션에 넣기
    try:
        coll = bpy.data.collections.new("DuckContainer_Solo")
        bpy.context.scene.collection.children.link(coll)
        for o in objs:
            for c in list(o.blender_obj.users_collection):
                c.objects.unlink(o.blender_obj)
            coll.objects.link(o.blender_obj)
    except Exception as e:
        print("Collection link failed:", e)


    print("Duck + Glass container loaded. Objects:", [o.get_name() for o in objs])