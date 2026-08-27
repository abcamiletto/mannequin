"""Numpy-only rigid and skinned mannequins driven by SMPL-X motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from mannequin import _identity, _io, _rigid

Array = Any
Kind = Literal["armor", "wooden"]
Lod = Literal[0, 1, 2]

BODY_JOINTS = 21
HAND_JOINTS = 30
SHAPE_COEFFICIENTS = 10
SMPLX_BODY_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22)


@dataclass
class Pose:
    """SMPL-X body motion used by a mannequin.

    All rotations are axis-angle. Leading batch dimensions are allowed.
    """

    body: Float[Array, "*batch 21 3"]
    hands: Float[Array, "*batch 30 3"]
    root_rotation: Float[Array, "*batch 3"]
    pelvis_rotation: Float[Array, "*batch 3"]
    translation: Float[Array, "*batch 3"]

    def copy(self) -> Pose:
        return Pose(
            body=np.asarray(self.body).copy(),
            hands=np.asarray(self.hands).copy(),
            root_rotation=np.asarray(self.root_rotation).copy(),
            pelvis_rotation=np.asarray(self.pelvis_rotation).copy(),
            translation=np.asarray(self.translation).copy(),
        )


class Mannequin:
    """One mannequin design and body shape.

    Armor uses rigid links. Wooden mannequins use linear blend skinning.
    Shape coefficients change bone lengths and reshape geometry along each bone.
    """

    def __init__(
        self,
        kind: Kind = "armor",
        *,
        lod: Lod | None = None,
        shape: Float[Array, "10"] | None = None,
    ) -> None:
        if kind != "armor" and lod is not None:
            raise ValueError(f"The {kind} mannequin has one resolution; omit lod.")
        asset_lod = 0 if lod is None else lod
        self._kind = kind
        self._weights = _io.load(asset_lod, kind=kind)
        self._bind_skin()
        self._template = _identity.build_template(self._weights)
        self._calibration = _io.load_calibration()
        if self.skinned:
            self._skeleton_weights = _io.load(2, kind="armor")
            self._skeleton_template = _identity.build_template(self._skeleton_weights)
        self.reshape(np.zeros(SHAPE_COEFFICIENTS, np.float32) if shape is None else shape)

    @property
    def kind(self) -> Kind:
        return self._kind

    @property
    def skinned(self) -> bool:
        return self._weights.skin_weights is not None

    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self._shape.copy()

    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._weights.joint_names)

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(self._weights.parents)

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(self._weights.link_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.vertices.shape[0]

    def reshape(self, shape: Float[Array, "10"]) -> None:
        """Set the ten SMPL-X shape coefficients."""
        shape = np.asarray(shape, dtype=self._weights.vertices.dtype)
        if shape.shape != (SHAPE_COEFFICIENTS,):
            raise ValueError(f"shape must have shape ({SHAPE_COEFFICIENTS},), got {shape.shape}")
        self._shape = shape.copy()
        self._identity = self._prepare_shape()

    def rest_pose(
        self,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: Any = np.float32,
    ) -> Pose:
        """Return a mutable neutral pose."""
        return Pose(
            body=np.zeros((*batch_shape, BODY_JOINTS, 3), dtype=dtype),
            hands=np.zeros((*batch_shape, HAND_JOINTS, 3), dtype=dtype),
            root_rotation=np.zeros((*batch_shape, 3), dtype=dtype),
            pelvis_rotation=np.zeros((*batch_shape, 3), dtype=dtype),
            translation=np.zeros((*batch_shape, 3), dtype=dtype),
        )

    def vertices(self, pose: Pose) -> Float[Array, "*batch V 3"]:
        """Return posed vertices."""
        if self.skinned:
            return self._skin_vertices(self.joint_transforms(pose))
        return self._rigid_vertices(self.joint_transforms(pose))

    def joint_transforms(self, pose: Pose) -> Float[Array, "*batch J 4 4"]:
        """Return world transforms for every joint."""
        return self._joint_transforms(pose, self._identity["local_joint_offsets"])

    def _joint_transforms(self, pose: Pose, local_offsets: np.ndarray) -> np.ndarray:
        rotations = self._local_rotations(pose)
        root_rotation = SO3.convert(pose.root_rotation, src="axis_angle", dst="rotmat", xp=np)
        pelvis_rotation = SO3.convert(pose.pelvis_rotation, src="axis_angle", dst="rotmat", xp=np)
        root_offset = local_offsets[0]
        root_translation = np.squeeze(root_rotation @ root_offset[..., None], axis=-1) + pose.translation
        return _rigid.forward_skeleton_from_local_rotations(
            rotations,
            local_offsets=local_offsets,
            actuated_joint_indices=self._weights.actuated_joint_indices,
            parents=self._weights.parents,
            global_rotation=root_rotation @ pelvis_rotation,
            global_translation=root_translation,
        )

    def link_transforms(self, pose: Pose) -> Float[Array, "*batch L 4 4"]:
        """Return world transforms for every rigid link."""
        return self.joint_transforms(pose)[..., self._weights.link_joint_indices, :, :]

    def _prepare_shape(self) -> _identity.Identity:
        if not self.skinned:
            return _identity.prepare(self._weights, self._template, self._calibration, self._shape)

        local_offsets = _identity.prepare_skeleton(
            self._skeleton_weights,
            self._skeleton_template,
            self._calibration,
            self._shape,
        )
        identity: _identity.Identity = {"local_joint_offsets": local_offsets}
        rotations = np.repeat(
            np.eye(3, dtype=self._weights.vertices.dtype)[None],
            len(self._weights.parents),
            axis=0,
        )
        joints = _rigid.forward_skeleton_from_local_rotations(
            rotations[self._weights.actuated_joint_indices],
            local_offsets=local_offsets,
            actuated_joint_indices=self._weights.actuated_joint_indices,
            parents=self._weights.parents,
            global_translation=local_offsets[0],
        )
        identity["skin_local_transforms"] = _identity.skin_shape_transforms(
            self._weights.local_offsets,
            local_offsets,
            self._weights.parents,
        )
        source_offset = self._weights.skin_root_offset
        assert source_offset is not None
        base_vertices = self._map_skin_to_skeleton(joints, identity, source_offset)
        target_floor = (self._calibration.sole_y_rest + self._calibration.sole_y_dirs @ self._shape).min()
        root_offset = source_offset.copy()
        root_offset[1] += target_floor - base_vertices[:, 1].min()
        identity["skin_root_offset"] = root_offset
        vertices = self._map_skin_to_skeleton(joints, identity, root_offset)
        head_min = self._calibration.head_min_rest + self._calibration.head_min_dirs @ self._shape
        head_max = self._calibration.head_max_rest + self._calibration.head_max_dirs @ self._shape
        vertices = self._fit_skin_region(vertices, ("Head",), head_min, head_max)
        identity["skin_vertices"] = self._preserve_rigid_parts(vertices, joints, identity, root_offset)
        identity["skin_bind_positions"] = joints[:, :3, 3] + root_offset
        return identity

    def _preserve_rigid_parts(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
        identity: _identity.Identity,
        root_offset: np.ndarray,
    ) -> np.ndarray:
        assignments = self._weights.skin_rigid_joint_indices
        source_joints = self._weights.skin_source_joint_positions
        assert assignments is not None and source_joints is not None

        result = vertices.copy()
        pelvis = self.joint_names.index("Pelvis")
        chest = self.joint_names.index("Chest")
        source_axis = source_joints[chest] - source_joints[pelvis]
        target_axis = joints[chest, :3, 3] - joints[pelvis, :3, 3]
        transform = _identity.align_similarity(source_axis, target_axis)
        region = assignments >= 0
        relative = self._weights.vertices[region] - source_joints[pelvis]
        result[region] = relative @ transform.T + joints[pelvis, :3, 3] + root_offset
        return result

    def _fit_skin_region(
        self,
        vertices: np.ndarray,
        joint_names: tuple[str, ...],
        target_min: np.ndarray,
        target_max: np.ndarray,
    ) -> np.ndarray:
        joint_indices = self._skin_joint_indices
        skin_weights = self._skin_weights
        assert joint_indices is not None and skin_weights is not None
        joints = [self.joint_names.index(name) for name in joint_names]
        influence = np.sum(np.where(np.isin(joint_indices, joints), skin_weights, 0.0), axis=1)
        region = vertices[influence > 0.5]
        source_min = region.min(axis=0)
        source_max = region.max(axis=0)
        scale = (target_max - target_min) / (source_max - source_min)
        fitted = target_min + (vertices - source_min) * scale
        return vertices + influence[:, None] * (fitted - vertices)

    def _bind_skin(self) -> None:
        self._skin_joint_indices = self._weights.skin_joint_indices
        self._skin_weights = self._weights.skin_weights
        if self._skin_joint_indices is None:
            assert self._skin_weights is None
            return
        assert self._skin_weights is not None
        influences = min(4, self._skin_weights.shape[1])
        strongest = np.argsort(self._skin_weights, axis=1)[:, -influences:]
        self._skin_joint_indices = np.take_along_axis(self._skin_joint_indices, strongest, axis=1)
        self._skin_weights = np.take_along_axis(self._skin_weights, strongest, axis=1)
        self._skin_weights /= self._skin_weights.sum(axis=1, keepdims=True)
        assignments = self._weights.skin_rigid_joint_indices
        assert assignments is not None
        rigid = assignments >= 0
        self._skin_joint_indices[rigid] = assignments[rigid, None]
        self._skin_weights[rigid] = 0.0
        self._skin_weights[rigid, 0] = 1.0

    def _local_rotations(self, pose: Pose) -> np.ndarray:
        body = np.asarray(pose.body)
        hands = np.asarray(pose.hands)
        if body.shape[-2:] != (BODY_JOINTS, 3):
            raise ValueError(f"pose.body must end in ({BODY_JOINTS}, 3), got {body.shape}")
        if hands.shape[-2:] != (HAND_JOINTS, 3):
            raise ValueError(f"pose.hands must end in ({HAND_JOINTS}, 3), got {hands.shape}")
        padding = np.zeros((*body.shape[:-2], 2, 3), dtype=body.dtype)
        ordered_body = np.take(np.concatenate((body, padding), axis=-2), SMPLX_BODY_ORDER, axis=-2)
        axis_angle = np.concatenate((ordered_body, hands), axis=-2)
        return SO3.convert(axis_angle, src="axis_angle", dst="rotmat", xp=np)

    def _rigid_vertices(self, joints: np.ndarray) -> np.ndarray:
        local_vertices = self._identity["link_local_vertices"]
        transforms = joints[..., self._weights.link_joint_indices, :, :]
        parts = []
        for transform, start, count in zip(
            np.moveaxis(transforms, -3, 0),
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            strict=True,
        ):
            vertices = local_vertices[start : start + count]
            parts.append(np.einsum("...ij,vj->...vi", transform[..., :3, :3], vertices) + transform[..., None, :3, 3])
        return np.concatenate(parts, axis=-2)

    def _skin_vertices(
        self,
        joints: np.ndarray,
    ) -> np.ndarray:
        joint_indices = self._skin_joint_indices
        skin_weights = self._skin_weights
        assert joint_indices is not None and skin_weights is not None
        relative = self._identity["skin_vertices"][:, None, :] - self._identity["skin_bind_positions"][joint_indices]
        rotations = joints[..., joint_indices, :3, :3]
        root_offset = self._identity["skin_root_offset"]
        rotated_offset = np.einsum("...ij,j->...i", joints[..., 0, :3, :3], root_offset)
        translations = joints[..., joint_indices, :3, 3] + rotated_offset[..., None, None, :]
        vertices = np.einsum("...vkij,vkj->...vki", rotations, relative) + translations
        return np.sum(vertices * skin_weights[..., None], axis=-2)

    def _map_skin_to_skeleton(
        self,
        joints: np.ndarray,
        identity: _identity.Identity,
        root_offset: np.ndarray,
    ) -> np.ndarray:
        joint_indices = self._skin_joint_indices
        skin_weights = self._skin_weights
        source_joints = self._weights.skin_source_joint_positions
        assert joint_indices is not None and skin_weights is not None and source_joints is not None
        relative = self._weights.vertices[:, None, :] - source_joints[joint_indices]
        transforms = identity["skin_local_transforms"]
        relative = np.einsum("vkij,vkj->vki", transforms[joint_indices], relative)
        vertices = relative + joints[joint_indices, :3, 3]
        vertices = np.sum(vertices * skin_weights[..., None], axis=-2)
        return vertices + root_offset


__all__ = ["Kind", "Mannequin", "Pose"]
