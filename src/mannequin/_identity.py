"""Shape-dependent identities for the rigid SMPL-X mannequin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import numpy as np
from jaxtyping import Float

from mannequin import _io as io

Array = Any


class Identity(TypedDict):
    """Skeleton and geometry prepared for one body shape."""

    local_joint_offsets: Float[Array, "J 3"]
    link_local_vertices: NotRequired[Float[Array, "V 3"]]
    skin_vertices: NotRequired[Float[Array, "V 3"]]
    skin_bind_positions: NotRequired[Float[Array, "J 3"]]
    skin_local_transforms: NotRequired[Float[Array, "J 3 3"]]


@dataclass(frozen=True)
class IdentityTemplate:
    """Neutral-shape quantities reused by every :func:`prepare` call."""

    rest_joints: np.ndarray
    joint_geom_anchors: tuple[int, ...]


def build_template(weights: io.MannequinWeights) -> IdentityTemplate:
    rest_joints = _joints_from_offsets(weights.local_offsets, weights.parents)
    # ball parts follow their owning joint, except balls that sit on a child
    # joint of their owner: shoulder balls (owned by the collars) and knuckle
    # balls (owned by the parent-side finger segment)
    anchors = list(weights.link_joint_indices)
    for link, name in enumerate(weights.link_names):
        if "shoulder_ball_L" in name:
            anchors[link] = weights.joint_names.index("L_Shoulder")
        elif "shoulder_ball_R" in name:
            anchors[link] = weights.joint_names.index("R_Shoulder")
        elif "_knuckle" in name:
            finger, knuckle, side = name.split("__")[1].split("_")
            anchors[link] = weights.joint_names.index(f"{side}_{finger.capitalize()}{knuckle[-1]}")
    return IdentityTemplate(rest_joints=rest_joints, joint_geom_anchors=tuple(anchors))


def prepare(
    weights: io.MannequinWeights,
    template: IdentityTemplate,
    calibration: io.ShapeCalibration,
    shape: np.ndarray,
) -> Identity:
    """Prepare rigid geometry for one SMPL-X body shape."""
    local_offsets = prepare_skeleton(weights, calibration, shape)
    rest_joints = _joints_from_offsets(local_offsets, weights.parents)
    display_joints = symmetric_joints(rest_joints, weights.joint_names)
    display_offsets = _offsets_from_joints(display_joints, weights.parents)
    local_vertices = _shape_vertices(
        weights,
        template,
        display_offsets,
        display_joints,
        rest_joints,
    )
    return {"local_joint_offsets": local_offsets, "link_local_vertices": local_vertices}


def prepare_skeleton(
    weights: io.MannequinWeights,
    calibration: io.ShapeCalibration,
    shape: np.ndarray,
) -> np.ndarray:
    """Prepare the calibrated skeleton for one SMPL-X body shape."""
    shape = np.asarray(shape)
    if shape.ndim != 1:
        raise ValueError(f"shape must have shape [S], got {shape.shape}.")

    shaped_joints = calibration.joint_rest + calibration.joint_dirs @ shape
    return _offsets_from_joints(shaped_joints, weights.parents).astype(weights.local_offsets.dtype)


def _offsets_from_joints(joints: np.ndarray, parents: list[int]) -> np.ndarray:
    offsets = np.zeros_like(joints)
    offsets[0] = joints[0]
    for joint in range(1, len(parents)):
        offsets[joint] = joints[joint] - joints[parents[joint]]
    return offsets


def symmetric_joints(joints: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Mirror joint pairs around x=0 without changing their shared midpoint."""
    result = joints.copy()
    indices = {name: index for index, name in enumerate(joint_names)}
    reflection = np.asarray((-1.0, 1.0, 1.0), dtype=joints.dtype)
    for name, left in indices.items():
        if not name.startswith("L_"):
            if not name.startswith("R_"):
                result[left, 0] = 0.0
            continue
        right = indices[f"R_{name[2:]}"]
        result[left] = 0.5 * (joints[left] + reflection * joints[right])
        result[right] = reflection * result[left]
    return result


def _shape_vertices(
    weights: io.MannequinWeights,
    template: IdentityTemplate,
    display_offsets: np.ndarray,
    display_joints: np.ndarray,
    kinematic_joints: np.ndarray,
) -> np.ndarray:
    transforms = joint_shape_transforms(weights.local_offsets, display_offsets, weights.parents)
    rest_parts = []
    for link, (owner, start, count, name) in enumerate(
        zip(
            weights.link_joint_indices,
            weights.link_vertex_starts,
            weights.link_vertex_counts,
            weights.link_names,
            strict=True,
        )
    ):
        vertices = weights.vertices[start : start + count]
        if "__joint_" in name:
            anchor = template.joint_geom_anchors[link]
            neutral_anchor = template.rest_joints[anchor] - template.rest_joints[owner]
            shaped_anchor = display_joints[anchor] - display_joints[owner]
            vertices = vertices + shaped_anchor - neutral_anchor
        else:
            vertices = vertices @ transforms[owner].T
        rest_parts.append(vertices + display_joints[owner])
    _align_abdomen(weights, rest_parts)
    _symmetrize_lateral_parts(weights, rest_parts)
    local_parts = [
        vertices - kinematic_joints[owner]
        for vertices, owner in zip(rest_parts, weights.link_joint_indices, strict=True)
    ]
    return np.concatenate(local_parts)


def _symmetrize_lateral_parts(
    weights: io.MannequinWeights,
    rest_parts: list[np.ndarray],
) -> None:
    part_names = [name.split("__")[1] for name in weights.link_names]
    indices = {name: index for index, name in enumerate(part_names)}
    reflection = np.asarray((-1.0, 1.0, 1.0), dtype=weights.vertices.dtype)
    for name, left in indices.items():
        if "_L" not in name:
            continue
        right = indices[name.replace("_L", "_R")]
        rest_parts[left] = 0.5 * (rest_parts[left] + reflection * rest_parts[right])
        rest_parts[right] = reflection * rest_parts[left]


def _align_abdomen(
    weights: io.MannequinWeights,
    rest_parts: list[np.ndarray],
) -> None:
    links = {
        token: next(index for index, name in enumerate(weights.link_names) if token in name)
        for token in ("pelvis_shell", "abdomen_shell", "chest_shell")
    }
    pelvis = rest_parts[links["pelvis_shell"]]
    abdomen = rest_parts[links["abdomen_shell"]].copy()
    chest = rest_parts[links["chest_shell"]]

    horizontal = np.asarray((0, 2))
    source_min = abdomen[:, horizontal].min(axis=0)
    source_max = abdomen[:, horizontal].max(axis=0)
    target_min = 0.5 * (pelvis[:, horizontal].min(axis=0) + chest[:, horizontal].min(axis=0))
    target_max = 0.5 * (pelvis[:, horizontal].max(axis=0) + chest[:, horizontal].max(axis=0))
    abdomen[:, horizontal] = target_min + (abdomen[:, horizontal] - source_min) * (
        (target_max - target_min) / (source_max - source_min)
    )

    link = links["abdomen_shell"]
    rest_parts[link] = abdomen


def joint_shape_transforms(
    neutral_offsets: np.ndarray,
    target_offsets: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    transforms = np.repeat(np.eye(3, dtype=target_offsets.dtype)[None], len(parents), axis=0)
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)

    for joint, joint_children in enumerate(children):
        if joint_children:
            neutral_axes = neutral_offsets[joint_children].T
            target_axes = target_offsets[joint_children].T
            transforms[joint] += (target_axes - neutral_axes) @ np.linalg.pinv(neutral_axes)
    return transforms


def skin_shape_transforms(
    neutral_offsets: np.ndarray,
    target_offsets: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    """Fit stable skin bind transforms to a target skeleton."""
    transforms = np.repeat(np.eye(3, dtype=target_offsets.dtype)[None], len(parents), axis=0)
    children = [[] for _ in parents]
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)

    for joint, joint_children in enumerate(children):
        if len(joint_children) == 1:
            child = joint_children[0]
            source = neutral_offsets[child]
            target = target_offsets[child]
            transforms[joint] = _align_and_scale(source, target)
        elif len(joint_children) > 1:
            transforms[joint] = _fit_similarity(
                neutral_offsets[joint_children].T,
                target_offsets[joint_children].T,
            )
            continue
        elif not joint_children and parents[joint] >= 0:
            source = neutral_offsets[joint]
            target = target_offsets[joint]
        else:
            continue
        transforms[joint] = _align_and_scale(source, target)
    return transforms


def align_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align one vector to another with uniform scale."""
    source_length = np.linalg.norm(source)
    target_length = np.linalg.norm(target)
    if source_length == 0.0 or target_length == 0.0:
        return np.eye(3, dtype=source.dtype)
    left, _, right = np.linalg.svd(_align_and_scale(source, target))
    return (left @ right) * (target_length / source_length)


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(target @ source.T)
    reflection = np.eye(3, dtype=source.dtype)
    reflection[-1, -1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ reflection @ right
    scale = np.sum(target * (rotation @ source)) / np.sum(source * source)
    return rotation * scale


def _align_and_scale(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_length = np.linalg.norm(source)
    target_length = np.linalg.norm(target)
    if source_length == 0.0 or target_length == 0.0:
        return np.eye(3, dtype=source.dtype)

    source_axis = source / source_length
    target_axis = target / target_length
    cross = np.cross(source_axis, target_axis)
    cosine = float(np.dot(source_axis, target_axis))
    cross_matrix = np.asarray(
        ((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)),
        dtype=source.dtype,
    )
    if cosine > -0.999999:
        rotation = np.eye(3, dtype=source.dtype) + cross_matrix + cross_matrix @ cross_matrix / (1.0 + cosine)
    else:
        basis = np.eye(3, dtype=source.dtype)[np.argmin(np.abs(source_axis))]
        axis = np.cross(source_axis, basis)
        axis /= np.linalg.norm(axis)
        rotation = 2.0 * np.outer(axis, axis) - np.eye(3, dtype=source.dtype)

    scale = np.eye(3, dtype=source.dtype)
    scale += (target_length / source_length - 1.0) * np.outer(source_axis, source_axis)
    return rotation @ scale


def _joints_from_offsets(offsets: np.ndarray, parents: list[int]) -> np.ndarray:
    joints = np.zeros_like(offsets)
    joints[0] = offsets[0]
    for joint in range(1, len(parents)):
        joints[joint] = joints[parents[joint]] + offsets[joint]
    return joints


__all__ = [
    "Identity",
    "IdentityTemplate",
    "align_similarity",
    "build_template",
    "joint_shape_transforms",
    "prepare",
    "prepare_skeleton",
    "skin_shape_transforms",
]
