"""Convert the wooden mannequin FBX into a compact skinned NPZ asset.

Run with Blender:

    blender --background --python authoring/import_wooden.py -- \
        wooden.fbx src/mannequin/assets/wooden.npz

The output retains the FBX vertices, triangles, skeleton binding, and vertex
weights.  Bone names are mapped to mannequin-x's equivalent SMPL-X joints.
"""

import hashlib
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

SOURCE_SHA256 = "4e1a4fc5b121d5fa61a631ee22ba360ca128279d794d1ed75b2acb9486e71cc8"
MAX_INFLUENCES = 8

argv = sys.argv[sys.argv.index("--") + 1 :]
fbx_path, output_path = map(Path, argv)
repo = Path(__file__).resolve().parents[1]
base_path = repo / "src" / "mannequin" / "assets" / "lod0.npz"
actual_sha256 = hashlib.sha256(fbx_path.read_bytes()).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(f"Unexpected FBX SHA256: {actual_sha256}")

bone_names = {
    "Spine1": "Torso",
    "Spine2": "Spine",
    "Spine3": "Chest",
    "L_Foot": "L_Toe",
    "R_Foot": "R_Toe",
    "L_Collar": "L_Thorax",
    "R_Collar": "R_Thorax",
}


def target_name(source_name):
    return bone_names.get(source_name, source_name)


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=str(fbx_path))
armature = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and len(obj.vertex_groups)]

# Blender imports the FBX as Z-up. mannequin-x and SMPL-X use Y-up.
zup_to_yup = Matrix(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
base = np.load(base_path, allow_pickle=False)
joint_names = base["joint_names"].tolist()
joint_index = {name: index for index, name in enumerate(joint_names)}

source_joint_positions = np.zeros((len(joint_names), 3), dtype=np.float32)
mapped_source_bones = {}
for bone in armature.data.bones:
    mapped = target_name(bone.name)
    if mapped not in joint_index:
        raise RuntimeError(f"Unmapped source bone: {bone.name}")
    mapped_source_bones[bone.name] = joint_index[mapped]
    source_head = zup_to_yup @ armature.matrix_world @ bone.head_local
    source_joint_positions[joint_index[mapped]] = source_head

# mannequin-x inserts zero-length hand joints between each wrist and its fingers.
for side in ("L", "R"):
    source_joint_positions[joint_index[f"{side}_Hand"]] = source_joint_positions[joint_index[f"{side}_Wrist"]]

packed_vertices = []
packed_faces = []
packed_joint_indices = []
packed_weights = []
part_names = []
part_vertex_starts = []
part_vertex_counts = []
part_face_starts = []
part_face_counts = []
vertex_cursor = 0
face_cursor = 0

for obj in meshes:
    mesh = obj.data
    mesh.calc_loop_triangles()
    group_names = {group.index: group.name for group in obj.vertex_groups}

    vertices = np.asarray([zup_to_yup @ obj.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float32)
    faces = np.asarray([triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64) + vertex_cursor
    indices = np.zeros((len(vertices), MAX_INFLUENCES), dtype=np.int16)
    weights = np.zeros((len(vertices), MAX_INFLUENCES), dtype=np.float32)

    for vertex in mesh.vertices:
        influences = sorted(vertex.groups, key=lambda item: item.weight, reverse=True)
        if len(influences) > MAX_INFLUENCES:
            raise RuntimeError(f"{obj.name} vertex {vertex.index} has {len(influences)} bone influences")
        for slot, influence in enumerate(influences):
            bone_name = group_names[influence.group]
            indices[vertex.index, slot] = mapped_source_bones[bone_name]
            weights[vertex.index, slot] = influence.weight
        total = weights[vertex.index].sum()
        if total == 0.0:
            raise RuntimeError(f"{obj.name} vertex {vertex.index} has no skin weights")
        weights[vertex.index] /= total

    packed_vertices.append(vertices)
    packed_faces.append(faces)
    packed_joint_indices.append(indices)
    packed_weights.append(weights)
    part_names.append(obj.name)
    part_vertex_starts.append(vertex_cursor)
    part_vertex_counts.append(len(vertices))
    part_face_starts.append(face_cursor)
    part_face_counts.append(len(faces))
    vertex_cursor += len(vertices)
    face_cursor += len(faces)

vertices = np.concatenate(packed_vertices)
faces = np.concatenate(packed_faces)
skin_joint_indices = np.concatenate(packed_joint_indices)
skin_weights = np.concatenate(packed_weights)

body = part_names.index("body")
body_start = part_vertex_starts[body]
body_count = part_vertex_counts[body]
body_faces = faces[part_face_starts[body] : part_face_starts[body] + part_face_counts[body]] - body_start
neighbors = [[] for _ in range(body_count)]
for a, b, c in body_faces:
    neighbors[a].extend((b, c))
    neighbors[b].extend((a, c))
    neighbors[c].extend((a, b))
components = []
remaining = set(range(body_count))
while remaining:
    component = {remaining.pop()}
    pending = list(component)
    while pending:
        for neighbor in neighbors[pending.pop()]:
            if neighbor in remaining:
                remaining.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)
    components.append(np.asarray(sorted(component), dtype=np.int64))
components.sort(key=lambda component: vertices[body_start + component, 1].mean(), reverse=True)
anchor_names = ("Chest", "Torso", "Pelvis")
if len(components) != len(anchor_names):
    raise RuntimeError(f"Expected three body shells, found {len(components)}")
skin_rigid_joint_indices = np.full(len(vertices), -1, dtype=np.int16)
for component, anchor_name in zip(components, anchor_names, strict=True):
    skin_rigid_joint_indices[body_start + component] = joint_index[anchor_name]

parents = base["parents"]
source_offsets = np.zeros_like(base["local_offsets"])
source_offsets[0] = base["local_offsets"][0]
for joint in range(1, len(parents)):
    source_offsets[joint] = source_joint_positions[joint] - source_joint_positions[parents[joint]]

values = {key: base[key] for key in base.files}
values.update(
    local_offsets=source_offsets,
    vertices=vertices,
    faces=faces,
    link_joint_indices=np.asarray([0], dtype=np.int64),
    link_vertex_starts=np.asarray([0], dtype=np.int64),
    link_vertex_counts=np.asarray([len(vertices)], dtype=np.int64),
    link_face_starts=np.asarray([0], dtype=np.int64),
    link_face_counts=np.asarray([len(faces)], dtype=np.int64),
    link_geom_positions=np.zeros((1, 3), dtype=np.float32),
    link_geom_rotations=np.eye(3, dtype=np.float32)[None],
    link_names=np.asarray(["Pelvis_J00__wooden__body"]),
    skin_joint_indices=skin_joint_indices,
    skin_weights=skin_weights,
    skin_source_joint_positions=source_joint_positions,
    skin_part_names=np.asarray(part_names),
    skin_part_vertex_starts=np.asarray(part_vertex_starts, dtype=np.int64),
    skin_part_vertex_counts=np.asarray(part_vertex_counts, dtype=np.int64),
    skin_part_face_starts=np.asarray(part_face_starts, dtype=np.int64),
    skin_part_face_counts=np.asarray(part_face_counts, dtype=np.int64),
    skin_rigid_joint_indices=skin_rigid_joint_indices,
    source_sha256=np.asarray(SOURCE_SHA256),
)
np.savez_compressed(output_path, **values)
print(f"WROTE {output_path} ({len(vertices)} vertices, {len(faces)} triangles)")
