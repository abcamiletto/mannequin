"""Shape-dependent identities for the rigid SMPL-X mannequin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

import numpy as np
from jaxtyping import Float

from mannequin import _io as io

Array = Any


class SmplxMannequinIdentity(TypedDict):
    """Skeleton and rigid geometry prepared once for one beta vector."""

    rest_joints: Float[Array, "J 3"]
    local_joint_offsets: Float[Array, "J 3"]
    rest_vertices: NotRequired[Float[Array, "V 3"]]
    link_local_vertices: NotRequired[Float[Array, "V 3"]]


@dataclass(frozen=True)
class IdentityTemplate:
    """Neutral-shape quantities reused by every :func:`prepare` call."""

    rest_joints: np.ndarray
    joint_geom_anchors: tuple[int, ...]


def build_template(weights: io.MannequinWeights) -> IdentityTemplate:
    rest_joints = _joints_from_offsets(weights.local_offsets, weights.parents)
    # ball parts follow their owning joint, except the shoulder balls which are
    # owned by the collar links but must track the shoulder joints
    anchors = list(weights.link_joint_indices)
    for link, name in enumerate(weights.link_names):
        if "shoulder_ball_L" in name:
            anchors[link] = weights.joint_names.index("L_Shoulder")
        elif "shoulder_ball_R" in name:
            anchors[link] = weights.joint_names.index("R_Shoulder")
    return IdentityTemplate(rest_joints=rest_joints, joint_geom_anchors=tuple(anchors))


def prepare(
    weights: io.MannequinWeights,
    template: IdentityTemplate,
    calibration: io.ShapeCalibration,
    shape: np.ndarray,
    *,
    skip_vertices: bool,
) -> SmplxMannequinIdentity:
    """Prepare a symmetric, length-only identity from one SMPL-X beta vector."""
    shape = np.asarray(shape)
    if shape.ndim != 1:
        raise ValueError(f"shape must have shape [S], got {shape.shape}.")

    local_offsets = weights.local_offsets.copy()
    if np.any(shape):
        shaped_joints = (calibration.joint_rest + calibration.joint_dirs @ shape).astype(local_offsets.dtype)
        measured_offsets = _offsets_from_joints(shaped_joints, weights.parents)
        local_offsets = _symmetric_length_offsets(weights.local_offsets, measured_offsets, weights.joint_names)
        rest_joints = _joints_from_offsets(local_offsets, weights.parents)
        _, mannequin_vertices = _shape_vertices(weights, template, local_offsets, rest_joints)
        smplx_floor = (calibration.sole_y_rest + calibration.sole_y_dirs @ shape).min()
        floor_correction = smplx_floor - mannequin_vertices[:, 1].min()
        ankle_indices = [weights.joint_names.index(name) for name in ("L_Ankle", "R_Ankle")]
        local_offsets[ankle_indices, 1] += floor_correction

    rest_joints = _joints_from_offsets(local_offsets, weights.parents)
    identity: SmplxMannequinIdentity = {
        "rest_joints": rest_joints,
        "local_joint_offsets": local_offsets,
    }
    if skip_vertices:
        return identity

    local_vertices, rest_vertices = _shape_vertices(weights, template, local_offsets, rest_joints)
    identity["link_local_vertices"] = local_vertices
    identity["rest_vertices"] = rest_vertices
    return identity


def _offsets_from_joints(joints: np.ndarray, parents: list[int]) -> np.ndarray:
    offsets = np.zeros_like(joints)
    offsets[0] = joints[0]
    for joint in range(1, len(parents)):
        offsets[joint] = joints[joint] - joints[parents[joint]]
    return offsets


def _symmetric_length_offsets(
    neutral_offsets: np.ndarray,
    measured_offsets: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    result = neutral_offsets.copy()
    result[0] = measured_offsets[0]
    indices = {name: index for index, name in enumerate(joint_names)}
    paired = set()
    reflection = np.array((-1.0, 1.0, 1.0), dtype=result.dtype)

    for name, left in indices.items():
        if not name.startswith("L_"):
            continue
        right = indices[f"R_{name[2:]}"]
        paired.update((left, right))
        neutral_length = np.linalg.norm(neutral_offsets[left])
        if neutral_length == 0.0:
            result[[left, right]] = 0.0
            continue
        length = 0.5 * (np.linalg.norm(measured_offsets[left]) + np.linalg.norm(measured_offsets[right]))
        result[left] *= length / neutral_length
        result[right] = result[left] * reflection

    for joint in range(1, len(result)):
        if joint in paired:
            continue
        neutral_length = np.linalg.norm(neutral_offsets[joint])
        if neutral_length > 0.0:
            result[joint] *= np.linalg.norm(measured_offsets[joint]) / neutral_length
    return result


def _shape_vertices(
    weights: io.MannequinWeights,
    template: IdentityTemplate,
    local_offsets: np.ndarray,
    rest_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms = _joint_transforms(weights.local_offsets, local_offsets, weights.parents)
    local_parts = []
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
            shaped_anchor = rest_joints[anchor] - rest_joints[owner]
            vertices = vertices + shaped_anchor - neutral_anchor
        else:
            vertices = vertices @ transforms[owner].T
        local_parts.append(vertices)
        rest_parts.append(vertices + rest_joints[owner])
    return np.concatenate(local_parts), np.concatenate(rest_parts)


def _joint_transforms(
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


def _joints_from_offsets(offsets: np.ndarray, parents: list[int]) -> np.ndarray:
    joints = np.zeros_like(offsets)
    joints[0] = offsets[0]
    for joint in range(1, len(parents)):
        joints[joint] = joints[parents[joint]] + offsets[joint]
    return joints


__all__ = ["IdentityTemplate", "SmplxMannequinIdentity", "build_template", "prepare"]
