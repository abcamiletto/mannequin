"""Native mannequin asset loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
from jaxtyping import Float, Int

Array = Any


@dataclass(frozen=True)
class ShapeCalibration:
    """Affine SMPL-X shape response baked in mannequin joint order.

    Rest joints are exactly linear in the first ten SMPL-X betas, and the
    shaped-body floor is the minimum over the (also linear) heights of the
    neutral sole vertices, so this small table reproduces the SMPL-X model's
    skeleton lengths and ground plane without the SMPL-X assets.
    """

    joint_rest: Float[Array, "J 3"]
    joint_dirs: Float[Array, "J 3 10"]
    sole_y_rest: Float[Array, "K"]
    sole_y_dirs: Float[Array, "K 10"]
    body_a_pose: Float[Array, "21 3"]
    hand_flat: Float[Array, "30 3"]
    hand_rest: Float[Array, "30 3"]


def load_calibration(*, dtype=np.float64) -> ShapeCalibration:
    """Load the baked SMPL-X shape calibration."""
    resource = files("mannequin") / "assets" / "smplx_calibration.npz"
    with resource.open("rb") as archive, np.load(archive, allow_pickle=False) as data:
        return ShapeCalibration(**{key: data[key].astype(dtype) for key in data.files})


@dataclass(frozen=True)
class MannequinWeights:
    joint_names: list[str]
    parents: list[int]
    local_offsets: Float[Array, "J 3"]
    vertices: Float[Array, "V 3"]
    faces: Int[Array, "F 3"]
    link_joint_indices: list[int]
    link_vertex_starts: list[int]
    link_vertex_counts: list[int]
    link_face_starts: list[int]
    link_face_counts: list[int]
    link_names: list[str]
    actuated_joint_indices: list[int]


def load(lod: int = 0, *, dtype=np.float32) -> MannequinWeights:
    """Load one bundled mannequin LOD."""
    if lod not in (0, 1, 2):
        raise ValueError(f"lod must be 0, 1, or 2; got {lod!r}")

    resource = files("mannequin") / "assets" / f"lod{lod}.npz"
    with resource.open("rb") as archive, np.load(archive, allow_pickle=False) as data:
        # the runtime assumes joint-local geometry: vertices are stored in their
        # owning joint's frame and joints carry no rest rotation
        joint_local = (
            np.allclose(data["rest_local_rotations"], np.eye(3), atol=1e-6)
            and np.allclose(data["link_geom_positions"], 0.0, atol=1e-9)
            and np.allclose(data["link_geom_rotations"], np.eye(3), atol=1e-9)
        )
        if not joint_local:
            raise ValueError(f"lod{lod}.npz geometry is not joint-local; rebuild the asset")
        return MannequinWeights(
            joint_names=data["joint_names"].tolist(),
            parents=data["parents"].tolist(),
            local_offsets=data["local_offsets"].astype(dtype),
            vertices=data["vertices"].astype(dtype),
            faces=data["faces"].astype(np.int64),
            link_joint_indices=data["link_joint_indices"].tolist(),
            link_vertex_starts=data["link_vertex_starts"].tolist(),
            link_vertex_counts=data["link_vertex_counts"].tolist(),
            link_face_starts=data["link_face_starts"].tolist(),
            link_face_counts=data["link_face_counts"].tolist(),
            link_names=data["link_names"].tolist(),
            actuated_joint_indices=data["actuated_joint_indices"].tolist(),
        )
