"""Native mannequin asset loading."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal

import numpy as np
from jaxtyping import Float, Int

Array = Any
Kind = Literal["armor", "wooden"]


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
    head_min_rest: Float[Array, "3"]
    head_max_rest: Float[Array, "3"]
    head_min_dirs: Float[Array, "3 10"]
    head_max_dirs: Float[Array, "3 10"]


def load_calibration(*, dtype=np.float64) -> ShapeCalibration:
    """Load the baked SMPL-X shape calibration."""
    resource = files("mannequin") / "assets" / "smplx_calibration.npz"
    with resource.open("rb") as archive, np.load(archive, allow_pickle=False) as data:
        return ShapeCalibration(
            joint_rest=data["joint_rest"].astype(dtype),
            joint_dirs=data["joint_dirs"].astype(dtype),
            sole_y_rest=data["sole_y_rest"].astype(dtype),
            sole_y_dirs=data["sole_y_dirs"].astype(dtype),
            head_min_rest=data["head_min_rest"].astype(dtype),
            head_max_rest=data["head_max_rest"].astype(dtype),
            head_min_dirs=data["head_min_dirs"].astype(dtype),
            head_max_dirs=data["head_max_dirs"].astype(dtype),
        )


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
    skin_joint_indices: Int[Array, "V K"] | None
    skin_weights: Float[Array, "V K"] | None
    skin_source_joint_positions: Float[Array, "J 3"] | None
    skin_root_offset: Float[Array, "3"] | None
    skin_part_names: list[str] | None
    skin_part_vertex_starts: list[int] | None
    skin_part_vertex_counts: list[int] | None
    skin_part_face_starts: list[int] | None
    skin_part_face_counts: list[int] | None
    skin_rigid_joint_indices: Int[Array, "V"] | None


def load(lod: int = 0, *, kind: Kind = "armor", dtype=np.float32) -> MannequinWeights:
    """Load one bundled mannequin style and LOD."""
    if lod not in (0, 1, 2):
        raise ValueError(f"lod must be 0, 1, or 2; got {lod!r}")
    if kind not in ("armor", "wooden"):
        raise ValueError(f"kind must be 'armor' or 'wooden'; got {kind!r}")

    asset_name = f"lod{lod}.npz" if kind == "armor" else f"{kind}.npz"
    resource = files("mannequin") / "assets" / asset_name
    with resource.open("rb") as archive, np.load(archive, allow_pickle=False) as data:
        skinned = "skin_weights" in data.files
        # the runtime assumes joint-local geometry: vertices are stored in their
        # owning joint's frame and joints carry no rest rotation
        joint_local = (
            np.allclose(data["rest_local_rotations"], np.eye(3), atol=1e-6)
            and np.allclose(data["link_geom_positions"], 0.0, atol=1e-9)
            and np.allclose(data["link_geom_rotations"], np.eye(3), atol=1e-9)
        )
        if not skinned and not joint_local:
            raise ValueError(f"{asset_name} geometry is not joint-local; rebuild the asset")
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
            skin_joint_indices=data["skin_joint_indices"].astype(np.int64) if skinned else None,
            skin_weights=data["skin_weights"].astype(dtype) if skinned else None,
            skin_source_joint_positions=data["skin_source_joint_positions"].astype(dtype) if skinned else None,
            skin_root_offset=data["skin_root_offset"].astype(dtype) if skinned else None,
            skin_part_names=data["skin_part_names"].tolist() if skinned else None,
            skin_part_vertex_starts=data["skin_part_vertex_starts"].tolist() if skinned else None,
            skin_part_vertex_counts=data["skin_part_vertex_counts"].tolist() if skinned else None,
            skin_part_face_starts=data["skin_part_face_starts"].tolist() if skinned else None,
            skin_part_face_counts=data["skin_part_face_counts"].tolist() if skinned else None,
            skin_rigid_joint_indices=data["skin_rigid_joint_indices"].astype(np.int64) if skinned else None,
        )
