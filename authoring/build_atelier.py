"""Polish an imported wooden NPZ in Blender and rebuild its shoulder sockets.

blender --background --factory-startup --python authoring/build_atelier.py -- \
    imported.npz src/mannequin/assets/atelier.npz authoring/atelier.blend

Use the unmodified output of import_wooden.py as input. The original head and
skeleton are preserved. Upper arms become turned links with spherical cups;
torso sockets follow the collar bones. All skin weights have at most four
normalized influences, matching the runtime.
"""

import itertools
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

PART_KEYS = (
    "skin_part_names",
    "skin_part_vertex_starts",
    "skin_part_vertex_counts",
    "skin_part_face_starts",
    "skin_part_face_counts",
)


def main():
    source, output, blend = map(Path, sys.argv[sys.argv.index("--") + 1 :])
    with np.load(source, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    if "polish_version" in values:
        raise ValueError("Input is already polished; rebuild from the original imported NPZ.")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = list(zip(*(values[key] for key in PART_KEYS), strict=True))
    polish_surfaces(values, parts)
    weights = values["skin_weights"]
    strongest = np.argsort(weights, axis=1)[:, -4:]
    values["skin_joint_indices"] = np.take_along_axis(values["skin_joint_indices"], strongest, axis=1)
    weights = np.take_along_axis(weights, strongest, axis=1)
    values["skin_weights"] = weights / weights.sum(axis=1, keepdims=True)
    parts = rebuild_upper_arms(values, parts)
    vertices = values["vertices"]
    bind_shoulders(values, parts)
    values["polish_version"] = np.asarray(2, dtype=np.int16)
    values["link_names"] = np.asarray(["Pelvis_J00__atelier__body"])
    assert np.isfinite(vertices).all()
    assert np.allclose(values["skin_weights"].sum(axis=1), 1.0)
    assert values["skin_weights"].shape == (len(vertices), 4)
    np.savez_compressed(output, **values)

    save_scene(values, parts, blend)
    print(f"WROTE {output}: {len(vertices)} vertices, four skin influences")


def components(faces, count):
    neighbors = [set() for _ in range(count)]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    remaining = set(range(count))
    while remaining:
        pending = [min(remaining)]
        remaining.remove(pending[0])
        found = []
        while pending:
            vertex = pending.pop()
            found.append(vertex)
            adjacent = neighbors[vertex] & remaining
            remaining.difference_update(adjacent)
            pending.extend(sorted(adjacent))
        yield np.asarray(sorted(found))


def smooth(vertices, faces):
    mesh = bpy.data.meshes.new("polish_temp")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # A light shrink-compensated pass removes lumpy highlights without
    # collapsing the spherical recesses or the finger separations.
    for _ in range(5):
        for factor in (0.38, -0.39):
            bmesh.ops.smooth_vert(
                bm,
                verts=list(bm.verts),
                factor=factor,
                use_axis_x=True,
                use_axis_y=True,
                use_axis_z=True,
            )
    bm.to_mesh(mesh)
    result = np.asarray([v.co[:] for v in mesh.vertices])
    bm.free()
    bpy.data.meshes.remove(mesh)
    return result


def slim_link(vertices, start, end, middle_scale, start_scale, end_scale):
    axis = end - start
    t = np.sum((vertices - start) * axis, axis=1) / np.sum(axis * axis)
    center = start + t[:, None] * axis
    u = np.clip(t, 0.0, 1.0)
    # Zero derivative at both socket mouths: no abrupt collar or pinching.
    envelope = np.sin(np.pi * u) ** 2
    scale = (start_scale * (1 - u) + end_scale * u) * (1 - envelope) + middle_scale * envelope
    return center + (vertices - center) * scale[:, None]


def zup(points):
    return np.stack((points[..., 0], -points[..., 2], points[..., 1]), axis=-1)


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def turned_upper_arm(start, end):
    """Closed turned link with concentric spherical cups and rounded lips."""
    axis = end - start
    length = np.linalg.norm(axis)
    axis /= length
    radial = np.cross(axis, (0, 0, 1))
    radial /= np.linalg.norm(radial)
    other = np.cross(axis, radial)
    shoulder_radius, elbow_radius = 0.0505, 0.0335
    shoulder_angle = np.arccos(0.020 / shoulder_radius)
    profile = [(shoulder_radius * np.cos(a), shoulder_radius * np.sin(a)) for a in np.linspace(0, shoulder_angle, 13)]
    # Hermite interpolation gives continuous slopes along the turned wood.
    stations = np.array([0.020, 0.024, 0.036, 0.075, 0.14, length - 0.05, length - 0.014])
    radii = np.array([profile[-1][1], 0.048, 0.0475, 0.042, 0.034, 0.030, 0.0315])
    slopes = np.gradient(radii, stations)
    slopes[0], slopes[-1] = 0.0, 0.0
    for i in range(len(stations) - 1):
        width = stations[i + 1] - stations[i]
        for t in np.linspace(0, 1, 7)[1:]:
            radius = (
                (2 * t**3 - 3 * t**2 + 1) * radii[i]
                + (t**3 - 2 * t**2 + t) * width * slopes[i]
                + (-2 * t**3 + 3 * t**2) * radii[i + 1]
                + (t**3 - t**2) * width * slopes[i + 1]
            )
            profile.append((stations[i] + width * t, radius))
    elbow_angle = np.arccos(0.014 / elbow_radius)
    profile += [(length - elbow_radius * np.cos(a), elbow_radius * np.sin(a)) for a in np.linspace(elbow_angle, 0, 13)]
    vertices, rings, faces = [], [], []
    for distance, radius in profile:
        ring = []
        angles = [0] if radius < 1e-8 else np.linspace(0, 2 * np.pi, 40, endpoint=False)
        for angle in angles:
            ring.append(len(vertices))
            vertices.append(start + distance * axis + radius * (np.cos(angle) * radial + np.sin(angle) * other))
        rings.append(ring)
    for a, b in itertools.pairwise(rings):
        if len(a) == 1:
            faces.extend((a[0], b[i], b[(i + 1) % len(b)]) for i in range(len(b)))
        elif len(b) == 1:
            faces.extend((a[i], b[0], a[(i + 1) % len(a)]) for i in range(len(a)))
        else:
            for i in range(len(a)):
                j = (i + 1) % len(a)
                faces.extend(((a[i], b[i], b[j]), (a[i], b[j], a[j])))
    mesh = bpy.data.meshes.new("Turned upper arm")
    mesh.from_pydata(np.asarray(vertices).tolist(), [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    result = np.asarray([v.co[:] for v in mesh.vertices]), np.asarray([p.vertices[:] for p in mesh.polygons])
    bm.free()
    bpy.data.meshes.remove(mesh)
    return result


def rebuild_upper_arms(values, parts):
    """Replace the hooked shoulder shells; keep all other parts and bindings."""
    packed = {
        key: [] for key in ("vertices", "faces", "skin_joint_indices", "skin_weights", "skin_rigid_joint_indices")
    }
    new_parts = []
    vertex_cursor = face_cursor = 0
    names = values["joint_names"].tolist()
    joints = values["skin_source_joint_positions"]
    for name, vs, vc, fs, fc in parts:
        vertices = values["vertices"][vs : vs + vc]
        faces = values["faces"][fs : fs + fc] - vs
        part_start, face_start = vertex_cursor, face_cursor
        for ids in components(faces, vc):
            owned = np.isin(faces[:, 0], ids)
            remap = np.zeros(vc, dtype=np.int64)
            remap[ids] = np.arange(len(ids))
            f = remap[faces[owned]]
            v = vertices[ids]
            indices = values["skin_joint_indices"][vs + ids]
            weights = values["skin_weights"][vs + ids]
            rigid = values["skin_rigid_joint_indices"][vs + ids]
            if name == "arm" and 0.2 < abs(v.mean(0)[0]) < 0.4:
                side = "L" if v.mean(0)[0] > 0 else "R"
                shoulder, elbow = names.index(f"{side}_Shoulder"), names.index(f"{side}_Elbow")
                v, f = turned_upper_arm(joints[shoulder].astype(float), joints[elbow].astype(float))
                indices = np.zeros((len(v), 4), dtype=np.int16)
                indices[:, 0] = shoulder
                weights = np.zeros((len(v), 4), dtype=np.float32)
                weights[:, 0] = 1
                rigid = np.full(len(v), -1, dtype=np.int16)
            for key, array in zip(packed, (v, f + vertex_cursor, indices, weights, rigid), strict=True):
                packed[key].append(array)
            vertex_cursor += len(v)
            face_cursor += len(f)
        new_parts.append((name, part_start, vertex_cursor - part_start, face_start, face_cursor - face_start))
    for key, arrays in packed.items():
        values[key] = np.concatenate(arrays).astype(values[key].dtype)
    for column, key in enumerate(PART_KEYS):
        values[key] = np.asarray([part[column] for part in new_parts], dtype=values[key].dtype)
    values["link_vertex_counts"] = np.asarray([vertex_cursor], dtype=np.int64)
    values["link_face_counts"] = np.asarray([face_cursor], dtype=np.int64)
    return new_parts


def polish_surfaces(values, parts):
    vertices = values["vertices"].astype(np.float64)
    original = vertices.copy()
    joints = dict(zip(values["joint_names"], values["skin_source_joint_positions"].astype(np.float64)))
    ball_scales = {"1": 0.90, "2": 0.93, "3": 0.85, "4": 0.86, "5": 0.92, "6": 0.92}
    for name, vs, vc, fs, fc in parts:
        faces = values["faces"][fs : fs + fc] - vs
        p = original[vs : vs + vc].copy() if name == "head" else smooth(original[vs : vs + vc], faces)
        for ids in components(faces, vc):
            q = p[ids]
            center = q.mean(axis=0)
            side = "L" if center[0] > 0 else "R"
            if name.startswith("joint_"):
                # Scale about the existing ball center; paired topology remains intact.
                q = center + (q - center) * ball_scales[name[-1]]
            elif name == "arm" and 0.44 < abs(center[0]) < 0.70:
                start, end = joints[f"{side}_Elbow"], joints[f"{side}_Wrist"]
                q = slim_link(q, start, end, 0.84, 0.90, 0.93)
            elif name == "leg" and center[1] <= -1.04:
                # Preserve the heel, toe, and sole contact surface exactly.
                q = original[vs + ids].copy()
            elif name == "leg":
                upper = center[1] > -0.67
                a, b = ("Hip", "Knee") if upper else ("Knee", "Ankle")
                q = slim_link(
                    q,
                    joints[f"{side}_{a}"],
                    joints[f"{side}_{b}"],
                    0.76 if upper else 0.83,
                    0.86 if upper else 0.92,
                    0.92,
                )
            elif name == "body":
                # Draw in the waist, ribcage depth, and pelvis, then fit the
                # socket neighborhoods about the same centers as their balls.
                before = q.copy()
                y = q[:, 1]
                neck = np.clip((y - 0.23) / 0.07, 0, 1)
                q[:, 0] *= 0.88 + 0.12 * neck
                q[:, 2] = center[2] + (q[:, 2] - center[2]) * (0.83 + 0.17 * neck)
                for suffix, scale, inner, outer in (
                    ("Shoulder", 0.85, 0.065, 0.12),
                    ("Hip", 0.86, 0.08, 0.14),
                ):
                    for socket_side in ("L", "R"):
                        pivot = joints[f"{socket_side}_{suffix}"]
                        distance = np.linalg.norm(before - pivot, axis=1)
                        t = np.clip((outer - distance) / (outer - inner), 0, 1)
                        blend_weight = (t * t * (3 - 2 * t))[:, None]
                        fitted = pivot + (before - pivot) * scale
                        q = q * (1 - blend_weight) + fitted * blend_weight
            p[ids] = q
        vertices[vs : vs + vc] = p

    values["vertices"] = vertices.astype(np.float32)


def bind_shoulders(values, parts):
    vertices = values["vertices"]
    # Let the torso socket follow the collar bone; keep the central chest rigid.
    # A full-weight inner ring maintains its fit as the clavicle raises or rolls.
    joint_names = values["joint_names"].tolist()
    body = next(part for part in parts if part[0] == "body")
    _, start, count, _, _ = body
    ids = np.arange(start, start + count)
    chest = joint_names.index("Chest")
    chest_ids = ids[values["skin_rigid_joint_indices"][ids] == chest]
    for side in ("L", "R"):
        shoulder = joint_names.index(f"{side}_Shoulder")
        collar = joint_names.index(f"{side}_Thorax")
        distance = np.linalg.norm(vertices[chest_ids] - values["skin_source_joint_positions"][shoulder], axis=1)
        t = np.clip((0.145 - distance) / (0.145 - 0.07), 0, 1)
        influence = t * t * (3 - 2 * t)
        selected = chest_ids[influence > 0]
        influence = influence[influence > 0]
        values["skin_rigid_joint_indices"][selected] = -1
        values["skin_joint_indices"][selected] = (chest, collar, 0, 0)
        values["skin_weights"][selected] = np.stack(
            (1 - influence, influence, np.zeros_like(influence), np.zeros_like(influence)),
            axis=1,
        )


def save_scene(values, parts, blend):
    vertices = values["vertices"]
    wood = material("Satin beech", (0.53, 0.32, 0.145), 0.28)
    joint_mat = material("End grain joints", (0.34, 0.185, 0.075), 0.34)
    # Store packed vertex ranges and weights on editable meshes.
    meshes = []
    for name, vs, vc, fs, fc in parts:
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(zup(vertices[vs : vs + vc]).tolist(), [], (values["faces"][fs : fs + fc] - vs).tolist())
        mesh.shade_smooth()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.data.materials.append(joint_mat if name.startswith("joint_") else wood)
        obj["npz_vertex_start"] = int(vs)
        groups = [obj.vertex_groups.new(name=str(n)) for n in values["joint_names"]]
        for i in range(vc):
            rigid = values["skin_rigid_joint_indices"][vs + i]
            if rigid >= 0:
                groups[rigid].add([i], 1.0, "REPLACE")
            else:
                for j, w in zip(values["skin_joint_indices"][vs + i], values["skin_weights"][vs + i]):
                    if w > 0:
                        groups[j].add([i], float(w), "REPLACE")
        meshes.append(obj)
    armature = bpy.data.armatures.new("Atelier skeleton")
    rig = bpy.data.objects.new("Atelier skeleton", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    positions = zup(values["skin_source_joint_positions"])
    bones = []
    for i, name in enumerate(values["joint_names"]):
        bone = armature.edit_bones.new(str(name))
        bone.head = positions[i]
        bone.tail = positions[i] + np.array((0, 0, 0.025))
        parent = int(values["parents"][i])
        if i and parent >= 0:
            bone.parent = bones[parent]
        bones.append(bone)
    bpy.ops.object.mode_set(mode="OBJECT")
    for obj in meshes:
        mod = obj.modifiers.new("Skin binding", "ARMATURE")
        mod.object = rig
    rig.hide_render = True
    rig.show_in_front = True
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("Studio")
    scene.world.color = (0.3, 0.3, 0.3)
    for name, position, energy, size in (
        ("Key", (-3, -4, 4), 650, 4),
        ("Fill", (3, -2, 1), 300, 3),
        ("Rim", (0, 2, 3), 450, 2),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        obj.rotation_euler = (Vector((0, 0, -0.25)) - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.collection.objects.link(camera)
    camera.location = (2.4, -4.5, 0.7)
    camera.rotation_euler = (Vector((0, 0, -0.3)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type, camera.data.ortho_scale = "ORTHO", 2.3
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.render.resolution_x, scene.render.resolution_y = 1200, 1200
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    meshes[0].select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.wm.save_as_mainfile(filepath=str(blend.resolve()))


if __name__ == "__main__":
    main()
