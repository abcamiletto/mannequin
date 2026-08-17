"""Numpy forward kinematics for the rigid mannequin."""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float, Int
from trimesh import Trimesh

Array = Any


def affine_transforms(
    rotations: Float[Array, "... 3 3"],
    translations: Float[Array, "... 3"],
) -> Float[Array, "... 4 4"]:
    """Assemble 4x4 transforms from rotations and translations."""
    batch = np.broadcast_shapes(rotations.shape[:-2], translations.shape[:-1])
    out = np.zeros((*batch, 4, 4), dtype=rotations.dtype)
    out[..., :3, :3] = rotations
    out[..., :3, 3] = translations
    out[..., 3, 3] = 1.0
    return out


def forward_skeleton_from_local_rotations(
    body_rotations: Float[Array, "... Q 3 3"],
    *,
    local_offsets: Float[Array, "J 3"],
    actuated_joint_indices: list[int],
    parents: list[int],
    global_translation: Float[Array, "... 3"] | None = None,
    global_rotation: Float[Array, "... 3 3"] | None = None,
) -> Float[Array, "... J 4 4"]:
    """Compute rigid hierarchy transforms from local actuated joint rotations."""
    batch_shape = tuple(body_rotations.shape[:-3])
    dtype = body_rotations.dtype
    num_joints = len(parents)

    local_rot = np.broadcast_to(np.eye(3, dtype=dtype), (*batch_shape, num_joints, 3, 3)).copy()
    local_rot[..., actuated_joint_indices, :, :] = body_rotations
    local_t = np.asarray(local_offsets, dtype=dtype)

    rot_world: list[Any] = [None] * num_joints
    pos_world: list[Any] = [None] * num_joints
    rot_world[0] = local_rot[..., 0, :, :]
    pos_world[0] = np.zeros((*batch_shape, 3), dtype=dtype)
    for joint in range(1, num_joints):
        parent = parents[joint]
        rot_world[joint] = rot_world[parent] @ local_rot[..., joint, :, :]
        pos_world[joint] = pos_world[parent] + np.squeeze(rot_world[parent] @ local_t[joint][..., None], axis=-1)

    rot = np.stack(rot_world, axis=-3)
    trans = np.stack(pos_world, axis=-2)
    if global_rotation is not None:
        rot = global_rotation[..., None, :, :] @ rot
        trans = np.squeeze(global_rotation[..., None, :, :] @ trans[..., None], axis=-1)
    if global_translation is not None:
        trans = trans + global_translation[..., None, :]

    return affine_transforms(rot, trans)


def link_meshes(
    vertices: Float[Array, "V 3"],
    faces: Int[Array, "F 3"],
    link_vertex_starts: list[int],
    link_vertex_counts: list[int],
    link_face_starts: list[int],
    link_face_counts: list[int],
) -> list[Trimesh]:
    """Build one link-local mesh per packed geometry range."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    meshes = []
    for vertex_start, vertex_count, face_start, face_count in zip(
        link_vertex_starts,
        link_vertex_counts,
        link_face_starts,
        link_face_counts,
        strict=True,
    ):
        link_vertices = vertices[vertex_start : vertex_start + vertex_count]
        link_faces = faces[face_start : face_start + face_count] - vertex_start
        meshes.append(Trimesh(vertices=link_vertices, faces=link_faces, process=False))
    return meshes
