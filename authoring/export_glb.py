"""Export one mannequin LOD as a rigged GLB for three.js and friends.

Usage: uv run python authoring/export_glb.py <lod> <out.glb>

The node tree mirrors the runtime FK exactly: a `mannequin` root (rest pelvis
translation), joint nodes nested by the skeleton hierarchy with the rest
local offsets, and each link mesh parented to its owning joint. Posing is
plain local rotations on the named joint nodes; the SMPL-X parameter order is
embedded in the root node's glTF extras. glTF and SMPL-X are both y-up, so
vertices are copied verbatim. The export is validated against
``link_transforms`` for a test pose before the file is written.
"""

import json
import struct
import sys

import numpy as np
from nanomanifold import SO3

from mannequin import PALETTES, Mannequin

ROUGHNESS = {"armor": 0.55, "joint": 0.6}


def srgb_to_linear(color):
    out = []
    for channel in color:
        c = channel / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def local_rotations(skeleton, parents):
    """Recover per-joint local rotations from world transforms."""
    local = [skeleton[0, :3, :3]]
    for joint in range(1, len(parents)):
        local.append(skeleton[parents[joint], :3, :3].T @ skeleton[joint, :3, :3])
    return np.stack(local)


def probe_parameter_order(model):
    """Map each SMPL-X pose parameter row to the mannequin joint it rotates."""
    parents = model.parents
    marker = np.array([0.7, 0.0, 0.0], np.float32)
    rest = model.rest_pose()
    order = {}
    for name, parameter, rows in (("body", "body_pose", 21), ("hands", "hand_pose", 30)):
        joints = []
        for row in range(rows):
            pose = {key: value.copy() for key, value in rest.items()}
            pose[parameter][row] = marker
            skeleton = np.asarray(model.joint_transforms(pose))
            moved = np.where(~np.isclose(local_rotations(skeleton, parents), np.eye(3), atol=1e-6).all((1, 2)))[0]
            assert len(moved) <= 1, (name, row, moved)
            joints.append(model.joint_names[int(moved[0])] if len(moved) else None)
        order[name] = joints
    return order


def build_glb(model, palette):
    weights = model._weights
    parents = model.parents
    armor_color, joint_color = palette

    blob = bytearray()
    buffer_views = []
    accessors = []

    def add_view(data):
        while len(blob) % 4:
            blob.append(0)
        buffer_views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes})
        blob.extend(data.tobytes())
        return len(buffer_views) - 1

    materials = [
        {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*srgb_to_linear(color), 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": ROUGHNESS[name],
            },
        }
        for name, color in (("armor", armor_color), ("joint", joint_color))
    ]

    meshes = []
    vertices = model._identity["link_local_vertices"]
    for link_name, vertex_start, vertex_count, face_start, face_count in zip(
        model.link_names,
        weights.link_vertex_starts,
        weights.link_vertex_counts,
        weights.link_face_starts,
        weights.link_face_counts,
        strict=True,
    ):
        positions = np.ascontiguousarray(vertices[vertex_start : vertex_start + vertex_count], dtype=np.float32)
        faces = weights.faces[face_start : face_start + face_count] - vertex_start
        indices = np.ascontiguousarray(faces, dtype=np.uint16).ravel()
        accessors.append(
            {
                "bufferView": add_view(positions),
                "componentType": 5126,
                "count": len(positions),
                "type": "VEC3",
                "min": positions.min(axis=0).tolist(),
                "max": positions.max(axis=0).tolist(),
            }
        )
        position_accessor = len(accessors) - 1
        accessors.append(
            {"bufferView": add_view(indices), "componentType": 5123, "count": len(indices), "type": "SCALAR"}
        )
        meshes.append(
            {
                "name": link_name.split("__")[1],
                "primitives": [
                    {
                        "attributes": {"POSITION": position_accessor},
                        "indices": len(accessors) - 1,
                        "material": 1 if "__joint_" in link_name else 0,
                    }
                ],
            }
        )

    offsets = np.asarray(model._identity["local_joint_offsets"], np.float64)
    rest_pelvis = offsets[0]

    # one node per joint (pelvis sits at the root origin), meshes as leaf children
    nodes = [{"name": name} for name in model.joint_names]
    for joint in range(1, len(model.joint_names)):
        nodes[joint]["translation"] = offsets[joint].tolist()
        nodes[parents[joint]].setdefault("children", []).append(joint)
    for link, glb_mesh in enumerate(meshes):
        nodes.append({"name": glb_mesh["name"], "mesh": link})
        nodes[weights.link_joint_indices[link]].setdefault("children", []).append(len(nodes) - 1)
    nodes.append(
        {
            "name": "mannequin",
            "translation": rest_pelvis.tolist(),
            "children": [0],
            "extras": {"smplx_order": probe_parameter_order(model), "rest_pelvis": rest_pelvis.tolist()},
        }
    )

    gltf = {
        "asset": {"version": "2.0", "generator": "mannequin-x export_glb"},
        "scene": 0,
        "scenes": [{"nodes": [len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode()
    json_bytes += b" " * (-len(json_bytes) % 4)
    while len(blob) % 4:
        blob.append(0)
    length = 12 + 8 + len(json_bytes) + 8 + len(blob)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, length),
            struct.pack("<II", len(json_bytes), 0x4E4F534A),
            json_bytes,
            struct.pack("<II", len(blob), 0x004E4942),
            bytes(blob),
        ]
    )


def validate(glb, model):
    """Re-parse the GLB and check nested FK against link_transforms."""
    json_length = struct.unpack_from("<I", glb, 12)[0]
    gltf = json.loads(glb[20 : 20 + json_length])
    nodes = gltf["nodes"]
    root = next(node for node in nodes if node["name"] == "mannequin")
    order = root["extras"]["smplx_order"]

    pose = model.rest_pose()
    pose["body_pose"][15, 2] = 0.5
    pose["hand_pose"][0, 0] = 0.3
    quats = {}
    for name, parameter in (("body", "body_pose"), ("hands", "hand_pose")):
        for joint_name, axis_angle in zip(order[name], pose[parameter], strict=True):
            if joint_name is not None:
                quats[joint_name] = SO3.convert(axis_angle, src="axis_angle", dst="quat", xp=np)

    world = {}
    link_world = {}

    def walk(index, rotation, position):
        node = nodes[index]
        position = position + rotation @ np.asarray(node.get("translation", [0.0, 0.0, 0.0]))
        if node["name"] in quats:
            rotation = rotation @ SO3.convert(quats[node["name"]], src="quat", dst="rotmat", xp=np)
        if "mesh" in node:
            link_world[node["mesh"]] = (rotation, position)
        world[node["name"]] = (rotation, position)
        for child in node.get("children", []):
            walk(child, rotation, position)

    walk(nodes.index(root), np.eye(3), np.zeros(3))

    expected = np.asarray(model.link_transforms(pose))
    for link in range(len(model.link_names)):
        rotation, position = link_world[link]
        np.testing.assert_allclose(rotation, expected[link, :3, :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(position, expected[link, :3, 3], atol=1e-5, rtol=0)


def main():
    lod, out_path = int(sys.argv[1]), sys.argv[2]
    model = Mannequin("armor", lod=lod)
    glb = build_glb(model, PALETTES["sand"])
    validate(glb, model)
    with open(out_path, "wb") as stream:
        stream.write(glb)
    print(f"WROTE {out_path} ({len(glb)} bytes, validated against link_transforms)")


if __name__ == "__main__":
    main()
