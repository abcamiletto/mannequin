"""Render the mannequin once per viser palette for a comparison sheet.

Usage: blender --background --python render_palettes.py -- <repo_dir> <out_dir>
"""

import json
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
REPO_DIR, OUT_DIR = argv[0], argv[1]
NPZ_PATH = f"{REPO_DIR}/src/mannequin/assets/lod0.npz"

# Palettes are defined in sRGB for viser; query them from the package itself.
palettes = json.loads(
    subprocess.run(
        [f"{REPO_DIR}/.venv/bin/python", "-c", "import json, mannequin; print(json.dumps(mannequin.PALETTES))"],
        check=True,
        capture_output=True,
        text=True,
        cwd="/tmp",
    ).stdout
)


def srgb_to_linear(color):
    out = []
    for channel in color:
        c = channel / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


STUDIO_COLOR = (0.34, 0.32, 0.29, 1.0)

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


def make_material(name, color, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    # Keep specular low: on dark palettes (charcoal) highlights otherwise dominate.
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    return m


mat_armor = make_material("Armor", (1, 1, 1, 1), 0.55)
mat_joint = make_material("Joints", (1, 1, 1, 1), 0.6)
mat_studio = make_material("Studio", STUDIO_COLOR, 0.78)

for link, name in enumerate(names):
    o = int(data["link_joint_indices"][link])
    vs, vc = int(data["link_vertex_starts"][link]), int(data["link_vertex_counts"][link])
    fs, fc = int(data["link_face_starts"][link]), int(data["link_face_counts"][link])
    world = verts[vs : vs + vc] + joints[o]
    world = np.stack([world[:, 0], -world[:, 2], world[:, 1]], axis=1)
    f = faces[fs : fs + fc] - vs
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(world.tolist(), [], f.tolist())
    mesh.validate()
    mesh.materials.append(mat_joint if "__joint_" in name else mat_armor)
    mesh.shade_smooth()
    if hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=0.9)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)

mn = np.array([1e9] * 3)
mx = -mn.copy()
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    for c in obj.bound_box:
        w = np.array(obj.matrix_world @ Vector(c))
        mn = np.minimum(mn, w)
        mx = np.maximum(mx, w)
center = (mn + mx) / 2
height = mx[2] - mn[2]

bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, mn[2] - 0.001))
bpy.context.active_object.data.materials.append(mat_studio)
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 6, mn[2] + 14), rotation=(1.5708, 0, 0))
bpy.context.active_object.data.materials.append(mat_studio)

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)
bg.inputs[1].default_value = 0.35


def add_light(name, loc, energy, size):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    scene.collection.objects.link(lo)
    direction = Vector((0, 0, center[2] + 0.2 * height)) - Vector(loc)
    lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


add_light("Key", (-2.7, -3.0, center[2] + 2.2), 900, 2.2)
add_light("Fill", (2.7, -3.0, center[2] + 2.2), 420, 2.6)
add_light("Rim", (0.0, 2.8, center[2] + 2.4), 650, 2.0)
add_light("Top", (0.0, 0.0, center[2] + 3.6), 260, 3.0)

span = float(max(mx[0] - mn[0], mx[2] - mn[2]))
d = span * 2.05
cd = bpy.data.cameras.new("cam")
cd.lens = 50
cam = bpy.data.objects.new("cam", cd)
cam.location = (d * 0.68, -d * 0.72, center[2] + 0.10 * height)
scene.collection.objects.link(cam)
direction = Vector((0, 0, center[2])) - Vector(cam.location)
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
scene.camera = cam

scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
gpu_found = False
for dev in prefs.devices:
    dev.use = dev.type in {"OPTIX", "CUDA"}
    gpu_found = gpu_found or dev.use
scene.cycles.device = "GPU" if gpu_found else "CPU"
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Base Contrast"
scene.render.resolution_x, scene.render.resolution_y = 900, 1400

for palette_name, (armor, joint) in palettes.items():
    mat_armor.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = srgb_to_linear(armor)
    mat_joint.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = srgb_to_linear(joint)
    scene.render.filepath = f"{OUT_DIR}/palette_{palette_name}.png"
    bpy.ops.render.render(write_still=True)
    print("WROTE", scene.render.filepath)
