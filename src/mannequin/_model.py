"""Rigid SMPL-X mannequin model (numpy-only)."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int
from nanomanifold import SO3
from trimesh import Trimesh

from mannequin import _identity as identity_ops
from mannequin import _io as io
from mannequin import _rigid as rigid

Array = Any
MannequinLod = Literal[0, 1, 2]

# source row in [*body_pose rows, zero, zero] for each mannequin actuated body
# slot; the two zero rows stand in for SMPL-X joints the mannequin adds rigidly
SMPLX_BODY_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 14, 12, 15, 17, 19, 21, 13, 16, 18, 20, 22)


class SmplxMannequin:
    """Rigid mannequin with SMPL-X pose and shape parameters.

    Shape changes alter symmetric bone lengths and reposition rigid parts; they
    never change link thickness or introduce skinning. All arrays are numpy.
    """

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 30
    NUM_HEAD_JOINTS = 3
    NUM_SHAPE_COEFFS = 10
    NUM_EXPR_COEFFS = 10

    def __init__(self, *, lod: MannequinLod = 0) -> None:
        self._weights = io.load(lod)
        self._identity_template = identity_ops.build_template(self._weights)
        self._calibration = io.load_calibration()

    # ------------------------------------------------------------- structure
    @property
    def faces(self) -> Int[Array, "F 3"]:
        return self._weights.faces

    @property
    def joint_names(self) -> list[str]:
        return list(self._weights.joint_names)

    @property
    def parents(self) -> list[int]:
        return list(self._weights.parents)

    @property
    def num_joints(self) -> int:
        return len(self._weights.parents)

    @property
    def num_vertices(self) -> int:
        return self._weights.vertices.shape[0]

    @property
    def link_names(self) -> list[str]:
        return list(self._weights.link_names)

    @cached_property
    def link_meshes(self) -> Sequence[Trimesh]:
        """Link-local meshes aligned with :attr:`link_names` and ``forward_links()``."""
        return rigid.link_meshes(
            self._weights.vertices,
            self._weights.faces,
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            self._weights.link_face_starts,
            self._weights.link_face_counts,
        )

    # --------------------------------------------------------------- forward
    def prepare_identity(
        self,
        shape: Float[Array, "10"],
        expression: Float[Array, "10"] | None = None,
        *,
        skip_vertices: bool = False,
    ) -> identity_ops.SmplxMannequinIdentity:
        """Prepare one reusable, length-only identity from an unbatched beta vector."""
        del expression  # the mannequin has no expression response
        return identity_ops.prepare(
            self._weights,
            self._identity_template,
            self._calibration,
            np.asarray(shape),
            skip_vertices=skip_vertices,
        )

    def forward_skeleton(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        head_pose: Float[Array, "*batch 3 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        shape: Float[Array, "10"] | None = None,
        expression: Float[Array, "10"] | None = None,
        identity: identity_ops.SmplxMannequinIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch J 4 4"]:
        """Compute shape-adjusted mannequin joint transforms."""
        identity = self._resolve_identity(identity, shape, expression, skip_vertices=True)
        root_translation = identity["local_joint_offsets"][0]
        rotation = None
        if global_rotation is not None:
            rotation = SO3.convert(global_rotation, src="axis_angle", dst="rotmat", xp=np)
            root_translation = np.squeeze(rotation @ root_translation[..., None], axis=-1)
        if global_translation is not None:
            root_translation = root_translation + global_translation
        if pelvis_rotation is not None:
            # the pelvis point is invariant under its own rotation, so pelvis
            # rotation folds into the global rotation without moving the root
            pelvis = SO3.convert(pelvis_rotation, src="axis_angle", dst="rotmat", xp=np)
            rotation = pelvis if rotation is None else rotation @ pelvis
        return rigid.forward_skeleton_from_local_rotations(
            self._local_rotations_from_smplx(body_pose, hand_pose),
            local_offsets=identity["local_joint_offsets"],
            actuated_joint_indices=self._weights.actuated_joint_indices,
            parents=self._weights.parents,
            global_rotation=rotation,
            global_translation=root_translation,
        )

    def forward_vertices(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        head_pose: Float[Array, "*batch 3 3"],
        hand_pose: Float[Array, "*batch 30 3"],
        *,
        pelvis_rotation: Float[Array, "*batch 3"] | None = None,
        shape: Float[Array, "10"] | None = None,
        expression: Float[Array, "10"] | None = None,
        identity: identity_ops.SmplxMannequinIdentity | None = None,
        global_rotation: Float[Array, "*batch 3"] | None = None,
        global_translation: Float[Array, "*batch 3"] | None = None,
    ) -> Float[Array, "*batch V 3"]:
        """Compute vertices for the shape-adjusted rigid parts."""
        identity = self._resolve_identity(identity, shape, expression, skip_vertices=False)
        skeleton = self.forward_skeleton(
            body_pose,
            head_pose,
            hand_pose,
            pelvis_rotation=pelvis_rotation,
            identity=identity,
            global_rotation=global_rotation,
            global_translation=global_translation,
        )
        return self._vertices_from_skeleton(skeleton, identity["link_local_vertices"])

    def forward_links(self, *args: Any, **kwargs: Any) -> Float[Array, "*batch L 4 4"]:
        """Compute link transforms from SMPL-X parameters."""
        # geometry is joint-local, so a link transform is its owning joint's
        skeleton = self.forward_skeleton(*args, **kwargs)
        return skeleton[..., self._weights.link_joint_indices, :, :]

    def forward_meshes(self, *args: Any, **kwargs: Any) -> list[Trimesh]:
        """Build one posed mannequin mesh per batch element."""
        vertices = np.asarray(self.forward_vertices(*args, **kwargs))
        if vertices.ndim == 2:
            vertices = vertices[None]
        vertices = vertices.reshape(-1, vertices.shape[-2], 3)
        return [Trimesh(vertices=item, faces=self.faces, process=False) for item in vertices]

    # ----------------------------------------------------------------- poses
    def get_rest_pose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X rest pose with configurable hand means."""
        if hands not in ("default", "flat", "rest"):
            raise ValueError(f"Invalid hands: {hands!r}")
        dtype = np.float32 if dtype is None else dtype
        shapes = {
            "shape": (self.NUM_SHAPE_COEFFS,),
            "expression": (self.NUM_EXPR_COEFFS,),
            "body_pose": (self.NUM_BODY_JOINTS, 3),
            "head_pose": (self.NUM_HEAD_JOINTS, 3),
            "hand_pose": (self.NUM_HAND_JOINTS, 3),
            "pelvis_rotation": (3,),
            "global_rotation": (3,),
            "global_translation": (3,),
        }
        params = {name: np.zeros((*batch_dims, *shape), dtype=dtype) for name, shape in shapes.items()}
        if hands != "default":
            hand_pose = np.asarray(getattr(self._calibration, f"hand_{hands}"), dtype=dtype)
            params["hand_pose"] = np.broadcast_to(hand_pose, (*batch_dims, *hand_pose.shape)).copy()
        return params

    def get_apose(
        self,
        *,
        batch_dims: tuple[int, ...] = (),
        dtype: Any | None = None,
        hands: Literal["default", "flat", "rest"] = "default",
    ) -> dict[str, Float[Array, "..."]]:
        """Return the SMPL-X A-pose."""
        params = self.get_rest_pose(batch_dims=batch_dims, dtype=dtype, hands=hands)
        body_pose = np.asarray(self._calibration.body_a_pose, dtype=params["body_pose"].dtype)
        params["body_pose"] = np.broadcast_to(body_pose, (*batch_dims, *body_pose.shape)).copy()
        return params

    # -------------------------------------------------------------- internals
    def _local_rotations_from_smplx(
        self,
        body_pose: Float[Array, "*batch 21 3"],
        hand_pose: Float[Array, "*batch 30 3"],
    ) -> Float[Array, "*batch Q 3 3"]:
        if body_pose.shape[-2:] != (self.NUM_BODY_JOINTS, 3):
            raise ValueError(f"body_pose must have shape [..., 21, 3], got {tuple(body_pose.shape)}")
        if hand_pose.shape[-2:] != (self.NUM_HAND_JOINTS, 3):
            raise ValueError(f"hand_pose must have shape [..., 30, 3], got {tuple(hand_pose.shape)}")
        padding = np.zeros((*body_pose.shape[:-2], 2, 3), dtype=body_pose.dtype)
        padded_body_pose = np.concatenate((body_pose, padding), axis=-2)
        ordered_body_pose = np.take(padded_body_pose, SMPLX_BODY_ORDER, axis=-2)
        axis_angle = np.concatenate((ordered_body_pose, hand_pose), axis=-2)
        return SO3.convert(axis_angle, src="axis_angle", dst="rotmat", xp=np)

    def _resolve_identity(
        self,
        identity: identity_ops.SmplxMannequinIdentity | None,
        shape: Float[Array, "10"] | None,
        expression: Float[Array, "10"] | None,
        *,
        skip_vertices: bool,
    ) -> identity_ops.SmplxMannequinIdentity:
        if identity is not None:
            conflicts = [name for name, value in (("shape", shape), ("expression", expression)) if value is not None]
            if conflicts:
                raise ValueError(f"identity cannot be combined with raw identity parameters: {', '.join(conflicts)}")
            return identity
        if shape is None:
            shape = np.zeros((self.NUM_SHAPE_COEFFS,), dtype=self._weights.vertices.dtype)
        return self.prepare_identity(shape, expression, skip_vertices=skip_vertices)

    def _vertices_from_skeleton(
        self,
        skeleton: Float[Array, "*batch J 4 4"],
        local_vertices: Float[Array, "V 3"],
    ) -> Float[Array, "*batch V 3"]:
        parts = []
        for owner, start, count in zip(
            self._weights.link_joint_indices,
            self._weights.link_vertex_starts,
            self._weights.link_vertex_counts,
            strict=True,
        ):
            rotation = skeleton[..., owner, :3, :3]
            translation = skeleton[..., owner, :3, 3]
            local = local_vertices[start : start + count]
            parts.append(np.squeeze(rotation[..., None, :, :] @ local[..., None], axis=-1) + translation[..., None, :])
        return np.concatenate(parts, axis=-2)


__all__ = ["SmplxMannequin"]
