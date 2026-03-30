"""
Blender headless inspector script to print where color data is stored in a .blend file.
Run with:
  blender --background --python inspect_blend.py -- /absolute/path/to/file.blend

It prints for each mesh object:
 - materials and their node setups (Image Texture names, links to Base Color)
 - material base color default value
 - whether the mesh has vertex color layers (and their names)
 - whether the mesh has color attributes (Blender 3.3+)
"""

import sys
import os

# Blender passes args after '--' into sys.argv; find that separator
if '--' in sys.argv:
    argv = sys.argv[sys.argv.index('--') + 1:]
else:
    argv = []

if len(argv) < 1:
    print('Usage: blender --background --python inspect_blend.py -- /path/to/file.blend')
    sys.exit(1)

blend_path = os.path.abspath(argv[0])
print(f"Inspecting: {blend_path}")

import bpy

# Open the blend file
bpy.ops.wm.open_mainfile(filepath=blend_path)

print('\nScene objects:')
for ob in bpy.data.objects:
    if ob.type != 'MESH':
        continue
    print('\n---')
    print(f"Object: {ob.name}")
    mesh = ob.data
    # Vertex colors (old API)
    vcol_names = []
    try:
        vcol_names = [vc.name for vc in mesh.vertex_colors]
    except Exception:
        vcol_names = []
    print('Vertex color layers (mesh.vertex_colors):', vcol_names)

    # New color attributes API (Blender 3.3+)
    attr_color_names = []
    try:
        attr_color_names = [a.name for a in mesh.color_attributes if a.domain == 'POINT' or a.domain == 'CORNER']
    except Exception:
        attr_color_names = []
    print('Color attributes (mesh.color_attributes):', attr_color_names)

    # Materials
    mats = ob.data.materials
    if len(mats) == 0:
        print('No materials assigned to object.')
    for i, m in enumerate(mats):
        if m is None:
            print(f'Material slot {i}: <None>')
            continue
        print(f'Material slot {i}: {m.name}')
        # check preview color / base color
        try:
            if m.use_nodes:
                print('  Uses nodes:')
                for node in m.node_tree.nodes:
                    ntype = node.type
                    # Image texture nodes
                    if ntype == 'TEX_IMAGE':
                        img = getattr(node, 'image', None)
                        print(f"    Image Texture node: {node.name} -> image={getattr(img,'name',None)} file='{getattr(img,'filepath',None)}'")
                    # Principled BSDF
                    if ntype == 'BSDF_PRINCIPLED':
                        bsdf = node
                        base_color = None
                        # find Base Color input link
                        link = bsdf.inputs['Base Color'].links
                        if link:
                            from_node = link[0].from_node
                            print(f"    Principled BSDF node: {node.name} -> Base Color linked from node {from_node.name} (type={from_node.type})")
                        else:
                            # default value
                            val = bsdf.inputs['Base Color'].default_value
                            print(f"    Principled BSDF node: {node.name} -> Base Color default = {val}")
            else:
                # non-node material: use diffuse color
                col = getattr(m, 'diffuse_color', None)
                print('  Non-node material. diffuse_color =', col)
        except Exception as e:
            print('  Error while inspecting material nodes:', e)

    # Also check vertex colors used in material (common pattern: Attribute Node or Vertex Color node)
    print('Searching materials node trees for Attribute/Vertex Color usage...')
    used_vcols = set()
    for m in mats:
        if m is None or not getattr(m, 'use_nodes', False):
            continue
        for node in m.node_tree.nodes:
            if node.type == 'ATTR':
                # Attribute node may reference a name
                name = getattr(node, 'attribute_name', None)
                print(f"  Attribute node '{node.name}' uses attribute '{name}'")
                used_vcols.add(name)
            if node.type == 'RGB':
                pass
            # In older Blender versions, there is a Vertex Color node named 'Vertex Color'
            if node.bl_idname == 'ShaderNodeVertexColor' or node.type == 'VCOL':
                vc_name = getattr(node, 'layer_name', None)
                print(f"  Vertex Color node '{node.name}' layer_name='{vc_name}'")
                used_vcols.add(vc_name)

    if used_vcols:
        print('Vertex color layers referenced by materials:', list(used_vcols))

print('\nDone.')
