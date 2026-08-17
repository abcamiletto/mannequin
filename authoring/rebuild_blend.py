"""Rebuild authoring/mannequin.blend from the packaged lod0 asset.

Usage: blender --background --python rebuild_blend.py -- <npz> <out_blend>

Scene: Z-up studio; each link is one object named like the blend convention
(J00__pelvis_shell__armor), origin at its owning joint, `smpl_joint_index`
custom property, armor/joint pastel materials, floor, camera, 3-point + top lights.
"""

import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
NPZ_PATH, OUT_BLEND = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

data = np.load(NPZ_PATH, allow_pickle=False)
parents = data["parents"]
offsets = data["local_offsets"].astype(np.float64)
joints = np.zeros_like(offsets)
joints[0] = offsets[0]
for j in range(1, len(parents)):
    joints[j] = joints[parents[j]] + offsets[j]

verts = data["vertices"].astype(np.float64)
faces = data["faces"]
names = data["link_names"].tolist()


def zup(v):
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def make_material(name, color, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    return m


mat_armor = make_material("Armor — edit ARMOR_COLOR", (0.68, 0.48, 0.24, 1.0), 0.3)
mat_joint = make_material("Joints — edit JOINT_COLOR", (0.52, 0.34, 0.16, 1.0), 0.38)
mat_studio = make_material("Studio", (0.34, 0.32, 0.29, 1.0), 0.78)

zmin = 1e9
for link, lname in enumerate(names):
    owner = int(data["link_joint_indices"][link])
    vs, vc = int(data["link_vertex_starts"][link]), int(data["link_vertex_counts"][link])
    fs, fc = int(data["link_face_starts"][link]), int(data["link_face_counts"][link])
    local = zup(verts[vs : vs + vc])
    f = faces[fs : fs + fc] - vs
    joint_name, middle, kind_tag = lname.split("__")
    kind = kind_tag.split("_")[0]
    jtag = joint_name.split("_J")[-1]
    obj_name = f"J{jtag}__{middle}__{kind}"
    mesh = bpy.data.meshes.new(obj_name)
    mesh.from_pydata(local.tolist(), [], f.tolist())
    mesh.validate()
    mesh.materials.append(mat_joint if kind == "joint" else mat_armor)
    mesh.shade_smooth()
    obj = bpy.data.objects.new(obj_name, mesh)
    obj.location = tuple(zup(joints[owner]))
    obj["smpl_joint_index"] = owner
    scene.collection.objects.link(obj)
    zmin = min(zmin, float(local[:, 2].min() + zup(joints[owner])[2]))

bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, zmin - 0.02))
floor = bpy.context.active_object
floor.name = "STUDIO_FLOOR"
floor.data.materials.append(mat_studio)

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35


def add_light(name, loc, energy, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    scene.collection.objects.link(lo)
    direction = Vector((0, 0, -0.4)) - Vector(loc)
    lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


add_light("Key", (-2.7, -3.0, 2.7), 900, 2.2)
add_light("Fill", (2.7, -3.0, 2.7), 420, 2.6)
add_light("Rim", (0.0, 2.8, 2.8), 650, 2.0)
add_light("Top", (0.0, 0.0, 4.0), 260, 3.0)

cam = bpy.data.cameras.new("Camera")
cam.lens = 50
co = bpy.data.objects.new("Camera", cam)
co.location = (0.0, -3.6, -0.35)
scene.collection.objects.link(co)
direction = Vector((0, 0, -0.44)) - co.location
co.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.camera = co
scene.render.resolution_x, scene.render.resolution_y = 1080, 1440

bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
print("SAVED", OUT_BLEND)
