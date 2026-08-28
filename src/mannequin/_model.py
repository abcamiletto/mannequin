"""Numpy-only rigid and skinned mannequins driven by SMPL-X motion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3

from mannequin import _identity, _io, _rigid

Array = Any
Kind = Literal["armor", "wooden"]
Lod = Literal[0, 1, 2]
PoseParameters = Mapping[str, Array]

BODY_JOINTS = 21
HAND_JOINTS = 30
SHAPE_COEFFICIENTS = 10
SMPLX_BODY_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22)
SMPLX_JOINT_NAMES = {
    "Torso": "Spine1",
    "Spine": "Spine2",
    "Chest": "Spine3",
    "L_Toe": "L_Foot",
    "R_Toe": "R_Foot",
    "L_Thorax": "L_Collar",
    "R_Thorax": "R_Collar",
}


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

    @property
    def body_pose(self) -> Array:
        return self.body

    @property
    def hand_pose(self) -> Array:
        return self.hands

    @property
    def global_rotation(self) -> Array:
        return self.root_rotation

    @property
    def global_translation(self) -> Array:
        return self.translation


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
        return tuple(SMPLX_JOINT_NAMES.get(name, name) for name in self._weights.joint_names)

    @property
    def parents(self) -> tuple[int, ...]:
        return tuple(self._weights.parents)

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(self._weights.link_names)

    @property
    def num_vertices(self) -> int:
        return self._weights.vertices.shape[0]

    def joint_index(self, joint: str) -> int:
        """Return a joint index using body-models SMPL-X names."""
        return self.joint_names.index(joint)

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

    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, np.ndarray]:
        """Return rest parameters with the same keys as body-models SMPL-X."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")
        dtype = np.float32 if dtype is None else dtype

        def zeros(*shape: int) -> np.ndarray:
            return np.zeros((*batch_dims, *shape), dtype=dtype)

        return {
            "shape": np.broadcast_to(self._shape, (*batch_dims, SHAPE_COEFFICIENTS)).astype(dtype, copy=True),
            "expression": zeros(10),
            "body_pose": zeros(BODY_JOINTS, 3),
            "head_pose": zeros(3, 3),
            "hand_pose": zeros(HAND_JOINTS, 3),
            "pelvis_rotation": zeros(3),
            "global_rotation": zeros(3),
            "global_translation": zeros(3),
        }

    def vertices(self, pose: Pose | PoseParameters) -> Float[Array, "*batch V 3"]:
        """Return posed vertices."""
        if not isinstance(pose, Pose):
            shape = self._shape_from_parameters(pose.get("shape"))
            if not np.allclose(shape, self._shape):
                raise ValueError("Pose shape differs from the mannequin; call reshape(shape[:10]) first.")
            pose = self._pose_from_parameters(pose)
        if self.skinned:
            return self._skin_vertices(self.joint_transforms(pose))
        return self._rigid_vertices(self.joint_transforms(pose))

    def joint_transforms(self, pose: Pose | PoseParameters) -> Float[Array, "*batch J 4 4"]:
        """Return world transforms for every joint."""
        if not isinstance(pose, Pose):
            return self.forward_skeleton(**pose)
        return self._joint_transforms(pose, self._identity["local_joint_offsets"])

    def forward_skeleton(
        self,
        body_pose: Array,
        head_pose: Array,
        hand_pose: Array,
        *,
        pelvis_rotation: Array | None = None,
        shape: Array | None = None,
        expression: Array | None = None,
        global_rotation: Array | None = None,
        global_translation: Array | None = None,
        joint_indices: Sequence[int] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute joints from the same arguments as body-models SMPL-X."""
        del expression
        head_pose = np.asarray(head_pose)
        if head_pose.shape[-2:] != (3, 3):
            raise ValueError(f"head_pose must end in (3, 3), got {head_pose.shape}")
        pose = self._pose_from_parameters(
            {
                "body_pose": body_pose,
                "hand_pose": hand_pose,
                "pelvis_rotation": pelvis_rotation,
                "global_rotation": global_rotation,
                "global_translation": global_translation,
            }
        )
        resolved_shape = self._shape_from_parameters(shape)
        local_offsets = _identity.prepare_skeleton(
            self._skeleton_weights if self.skinned else self._weights,
            self._calibration,
            resolved_shape,
        )
        joints = self._joint_transforms(pose, local_offsets)
        return joints if joint_indices is None else np.take(joints, joint_indices, axis=-3)

    def _pose_from_parameters(self, parameters: PoseParameters) -> Pose:
        body = np.asarray(parameters["body_pose"])
        batch_shape = body.shape[:-2]

        def zeros() -> np.ndarray:
            return np.zeros((*batch_shape, 3), dtype=body.dtype)

        def parameter(name: str) -> Array:
            value = parameters.get(name)
            return zeros() if value is None else value

        return Pose(
            body=body,
            hands=parameters["hand_pose"],
            root_rotation=parameter("global_rotation"),
            pelvis_rotation=parameter("pelvis_rotation"),
            translation=parameter("global_translation"),
        )

    def _shape_from_parameters(self, shape: Array | None) -> np.ndarray:
        if shape is None:
            return self._shape
        coefficients = np.asarray(shape)[..., :SHAPE_COEFFICIENTS].reshape(-1, SHAPE_COEFFICIENTS)
        if not np.allclose(coefficients, coefficients[0]):
            raise ValueError("Batched poses must use the same shape coefficients.")
        return coefficients[0]

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

    def link_transforms(self, pose: Pose | PoseParameters) -> Float[Array, "*batch L 4 4"]:
        """Return world transforms for every rigid link."""
        return self.joint_transforms(pose)[..., self._weights.link_joint_indices, :, :]

    def _prepare_shape(self) -> _identity.Identity:
        if not self.skinned:
            return _identity.prepare(self._weights, self._template, self._calibration, self._shape)

        local_offsets = _identity.prepare_skeleton(
            self._skeleton_weights,
            self._calibration,
            self._shape,
        )
        rest_joints = _identity._joints_from_offsets(local_offsets, self._weights.parents)
        display_joints = _identity.symmetric_joints(rest_joints, self._weights.joint_names)
        display_offsets = _identity._offsets_from_joints(display_joints, self._weights.parents)
        identity: _identity.Identity = {"local_joint_offsets": local_offsets}
        rotations = np.repeat(
            np.eye(3, dtype=self._weights.vertices.dtype)[None],
            len(self._weights.parents),
            axis=0,
        )
        joints = _rigid.forward_skeleton_from_local_rotations(
            rotations[self._weights.actuated_joint_indices],
            local_offsets=display_offsets,
            actuated_joint_indices=self._weights.actuated_joint_indices,
            parents=self._weights.parents,
            global_translation=display_offsets[0],
        )
        identity["skin_local_transforms"] = _identity.skin_shape_transforms(
            self._weights.local_offsets,
            display_offsets,
            self._weights.parents,
        )
        head = self._weights.joint_names.index("Head")
        identity["skin_local_transforms"][head] = np.eye(3, dtype=self._weights.vertices.dtype)
        vertices = self._map_skin_to_skeleton(joints, identity)
        head_min = self._calibration.head_min_rest + self._calibration.head_min_dirs @ self._shape
        head_max = self._calibration.head_max_rest + self._calibration.head_max_dirs @ self._shape
        vertices = self._fit_skin_region(vertices, ("Head",), head_min, head_max)
        bind_positions = rest_joints
        vertices = self._preserve_rigid_parts(vertices, joints)
        target_floor = (self._calibration.sole_y_rest + self._calibration.sole_y_dirs @ self._shape).min()
        vertices = self._reshape_forefeet(vertices, bind_positions, target_floor)
        identity["skin_vertices"] = self._symmetrize_skin_joints(vertices)
        identity["skin_bind_positions"] = bind_positions
        return identity

    def _symmetrize_skin_joints(self, vertices: np.ndarray) -> np.ndarray:
        names = self._weights.skin_part_names
        starts = self._weights.skin_part_vertex_starts
        counts = self._weights.skin_part_vertex_counts
        assert names is not None and starts is not None and counts is not None
        parts = {name: slice(start, start + count) for name, start, count in zip(names, starts, counts, strict=True)}
        reflection = np.asarray((-1.0, 1.0, 1.0), dtype=vertices.dtype)
        result = vertices.copy()
        for name, left in parts.items():
            if not name.startswith("joint_L"):
                continue
            right = parts[name.replace("joint_L", "joint_R")]
            result[left] = 0.5 * (vertices[left] + reflection * vertices[right])
            result[right] = reflection * result[left]
        return result

    def _reshape_forefeet(
        self,
        vertices: np.ndarray,
        bind_positions: np.ndarray,
        target_floor: float,
    ) -> np.ndarray:
        joint_indices = self._skin_joint_indices
        skin_weights = self._skin_weights
        assert joint_indices is not None and skin_weights is not None
        result = vertices.copy()
        angle = 0.1
        length_scale = 1.2
        cosine, sine = np.cos(angle), np.sin(angle)
        for side in ("L", "R"):
            ankle = self._weights.joint_names.index(f"{side}_Ankle")
            toe = self._weights.joint_names.index(f"{side}_Toe")
            foot_weight = np.sum(
                np.where(np.isin(joint_indices, (ankle, toe)), skin_weights, 0.0),
                axis=1,
            )
            relative = vertices - bind_positions[ankle]
            rotated = relative.copy()
            rotated[:, 1] = cosine * relative[:, 1] - sine * length_scale * relative[:, 2]
            rotated[:, 2] = sine * relative[:, 1] + cosine * length_scale * relative[:, 2]
            result += foot_weight[:, None] * (rotated - relative)

            foot = foot_weight > 0.5
            shift = np.max((target_floor - result[foot, 1]) / foot_weight[foot])
            result[:, 1] += foot_weight * shift
        return result

    def _preserve_rigid_parts(
        self,
        vertices: np.ndarray,
        joints: np.ndarray,
    ) -> np.ndarray:
        assignments = self._weights.skin_rigid_joint_indices
        source_joints = self._weights.skin_source_joint_positions
        assert assignments is not None and source_joints is not None

        result = vertices.copy()
        pelvis = self._weights.joint_names.index("Pelvis")
        chest = self._weights.joint_names.index("Chest")
        source_axis = source_joints[chest] - source_joints[pelvis]
        target_axis = joints[chest, :3, 3] - joints[pelvis, :3, 3]
        transform = _identity.align_similarity(source_axis, target_axis)
        region = assignments >= 0
        relative = self._weights.vertices[region] - source_joints[pelvis]
        result[region] = relative @ transform.T + joints[pelvis, :3, 3]
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
        joints = [self._weights.joint_names.index(name) for name in joint_names]
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
        translations = joints[..., joint_indices, :3, 3]
        vertices = np.einsum("...vkij,vkj->...vki", rotations, relative) + translations
        return np.sum(vertices * skin_weights[..., None], axis=-2)

    def _map_skin_to_skeleton(
        self,
        joints: np.ndarray,
        identity: _identity.Identity,
    ) -> np.ndarray:
        joint_indices = self._skin_joint_indices
        skin_weights = self._skin_weights
        source_joints = self._weights.skin_source_joint_positions
        assert joint_indices is not None and skin_weights is not None and source_joints is not None
        relative = self._weights.vertices[:, None, :] - source_joints[joint_indices]
        transforms = identity["skin_local_transforms"]
        relative = np.einsum("vkij,vkj->vki", transforms[joint_indices], relative)
        vertices = relative + joints[joint_indices, :3, 3]
        return np.sum(vertices * skin_weights[..., None], axis=-2)


__all__ = ["Kind", "Mannequin", "Pose"]
