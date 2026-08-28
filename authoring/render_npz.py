"""Render a mannequin NPZ asset in a studio scene.

Usage: blender --background --python render_npz.py -- <npz_path> <out_dir> <prefix>
"""

import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
NPZ_PATH, OUT_DIR, PREFIX = argv[0], argv[1], argv[2]

ARMOR_COLOR = (0.68, 0.48, 0.24, 1.0)
JOINT_COLOR = (0.16, 0.09, 0.05, 1.0)
STUDIO_COLOR = (0.34, 0.32, 0.29, 1.0)


def yup_to_zup(v):
    return (v[0], -v[2], v[1])


# ---- wipe scene ----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---- load asset ----
data = np.load(NPZ_PATH, allow_pickle=False)
parents = data["parents"]
offsets = data["local_offsets"].astype(np.float64)
joints = np.zeros_like(offsets)
joints[0] = offsets[0]
for j in range(1, len(parents)):
    joints[j] = joints[parents[j]] + offsets[j]

verts = data["vertices"].astype(np.float64)
faces = data["faces"]
skinned = "skin_weights" in data.files
if skinned:
    skin_joints = data["skin_joint_indices"]
    skin_weights = data["skin_weights"]
    source_joints = data["skin_source_joint_positions"]
    rest = np.sum(
        (verts[:, None, :] - source_joints[skin_joints] + joints[skin_joints]) * skin_weights[..., None], axis=1
    )
    world_vertices = np.stack([rest[:, 0], -rest[:, 2], rest[:, 1]], axis=1)
    names = data["skin_part_names"].tolist()
    vertex_starts = data["skin_part_vertex_starts"]
    vertex_counts = data["skin_part_vertex_counts"]
    face_starts = data["skin_part_face_starts"]
    face_counts = data["skin_part_face_counts"]
else:
    names = data["link_names"].tolist()
    vertex_starts = data["link_vertex_starts"]
    vertex_counts = data["link_vertex_counts"]
    face_starts = data["link_face_starts"]
    face_counts = data["link_face_counts"]


def make_material(name, color, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = rough
    return m


mat_armor = make_material("Armor", ARMOR_COLOR, 0.3)
mat_joint = make_material("Joints", JOINT_COLOR, 0.38)
mat_studio = make_material("Studio", STUDIO_COLOR, 0.78)

for link, name in enumerate(names):
    vs, vc = int(vertex_starts[link]), int(vertex_counts[link])
    fs, fc = int(face_starts[link]), int(face_counts[link])
    if skinned:
        world = world_vertices[vs : vs + vc]
    else:
        owner = int(data["link_joint_indices"][link])
        world = verts[vs : vs + vc] + joints[owner]
        world = np.stack([world[:, 0], -world[:, 2], world[:, 1]], axis=1)
    f = faces[fs : fs + fc] - vs
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(world.tolist(), [], f.tolist())
    mesh.validate()
    mesh.materials.append(mat_joint if name.startswith("joint_") or "__joint_" in name else mat_armor)
    mesh.shade_smooth()
    if hasattr(mesh, "set_sharp_from_angle"):
        mesh.set_sharp_from_angle(angle=0.9)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)

# ---- bounds ----
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

# head bounds (highest armor part)
head_obj = next((o for o in scene.objects if "head" in o.name), None)
if head_obj is None:
    hsize = height * 0.16
    hcenter = np.asarray((center[0], center[1], mx[2] - hsize * 0.5))
else:
    hmn = np.array([1e9] * 3)
    hmx = -hmn.copy()
    for c in head_obj.bound_box:
        w = np.array(head_obj.matrix_world @ Vector(c))
        hmn = np.minimum(hmn, w)
        hmx = np.maximum(hmx, w)
    hcenter = (hmn + hmx) / 2
    hsize = (hmx - hmn).max()

# ---- studio ----
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, mn[2] - 0.001))
floor = bpy.context.active_object
floor.data.materials.append(mat_studio)
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 6, mn[2] + 14), rotation=(1.5708, 0, 0))
wall = bpy.context.active_object
wall.data.materials.append(mat_studio)

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
    return lo


add_light("Key", (-2.7, -3.0, center[2] + 2.2), 900, 2.2)
add_light("Fill", (2.7, -3.0, center[2] + 2.2), 420, 2.6)
add_light("Rim", (0.0, 2.8, center[2] + 2.4), 650, 2.0)
add_light("Top", (0.0, 0.0, center[2] + 3.6), 260, 3.0)


def add_camera(name, loc, target, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    scene.collection.objects.link(co)
    direction = Vector(target) - Vector(loc)
    co.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return co


body_target = (0, 0, center[2])
span = float(max(mx[0] - mn[0], mx[2] - mn[2]))
d = span * 2.05
cams = {
    "full_front": (add_camera("c1", (0, -d, center[2] + 0.06 * height), body_target, 50), (1080, 1440)),
    "full_quarter": (add_camera("c2", (d * 0.68, -d * 0.72, center[2] + 0.10 * height), body_target, 50), (1080, 1440)),
    "face_front": (
        add_camera("c3", (0, hcenter[1] - hsize * 3.4, hcenter[2] + hsize * 0.1), tuple(hcenter), 85),
        (1200, 1200),
    ),
    "face_quarter": (
        add_camera("c4", (hsize * 2.4, hcenter[1] - hsize * 2.6, hcenter[2] + hsize * 0.5), tuple(hcenter), 85),
        (1200, 1200),
    ),
    "hand_left": (None, (1200, 1200)),
}

cams["feet"] = (
    add_camera("c6", (0.5, -0.62, mn[2] + 0.34), (0.0, -0.02, mn[2] + 0.08), 70),
    (1200, 1200),
)

# hand closeup: find left palm
palm = next((o for o in scene.objects if "palm_L" in o.name), None)
if palm is not None:
    pmn = np.array([1e9] * 3)
    pmx = -pmn.copy()
    for c in palm.bound_box:
        w = np.array(palm.matrix_world @ Vector(c))
        pmn = np.minimum(pmn, w)
        pmx = np.maximum(pmx, w)
    pc = (pmn + pmx) / 2
    cams["hand_left"] = (
        add_camera("c5", (pc[0] + 0.28, pc[1] - 0.42, pc[2] + 0.22), tuple(pc), 85),
        (1200, 1200),
    )
else:
    del cams["hand_left"]

# ---- render settings ----
scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.cycles.use_denoising = True
prefs = bpy.context.preferences.addons["cycles"].preferences
for backend in ("OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"):
    try:
        prefs.compute_device_type = backend
        break
    except TypeError:
        continue
prefs.get_devices()
gpu_found = False
for dev in prefs.devices:
    dev.use = dev.type != "CPU"
    gpu_found = gpu_found or dev.use
scene.cycles.device = "GPU" if gpu_found else "CPU"
print("RENDER DEVICE:", scene.cycles.device)
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Base Contrast"

for view, (cam, res) in cams.items():
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.filepath = f"{OUT_DIR}/{PREFIX}_{view}.png"
    bpy.ops.render.render(write_still=True)
    print("WROTE", scene.render.filepath)
