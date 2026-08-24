"""Viser integration for the rigid mannequin.

The mannequin has no skinning, so geometry is uploaded to the scene exactly
once: each link mesh sits under its owning joint's frame, and pose updates
only move the per-joint frame transforms (two messages per joint instead of
two per mesh, so a full pose fits in a single viser message window).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

from mannequin._model import SmplxMannequin

if TYPE_CHECKING:
    import viser

Array = Any

Color = tuple[int, int, int]

PALETTES: dict[str, tuple[Color, Color]] = {
    "sand": ((211, 151, 63), (133, 78, 20)),
    "ivory": ((245, 235, 210), (117, 91, 62)),
    "charcoal": ((42, 45, 53), (125, 138, 160)),
    "sage": ((126, 167, 111), (54, 96, 57)),
    "clay": ((211, 91, 57), (120, 48, 28)),
    "slate": ((77, 116, 161), (32, 59, 96)),
}
"""Predefined (armor, joint) RGB pairs for :func:`add_mannequin`."""


class ViserMannequinHandle:
    """Mannequin added to a viser scene as static meshes under per-joint frames.

    ``set_pose`` accepts the same parameter names as ``SmplxMannequin`` forward
    calls (so ``handle.set_pose(**params)`` works with full SMPL-X parameter
    dicts); ``shape`` entries are routed to :meth:`set_shape` and
    ``expression`` is ignored. Unchanged values are skipped, and meshes are
    never re-uploaded.
    """

    def __init__(
        self,
        model: SmplxMannequin,
        pose: dict[str, Float[np.ndarray, "..."]],
        root_frame: viser.FrameHandle,
        frames: list[viser.FrameHandle],
        frame_joints: list[int],
        links: list[viser.MeshHandle],
    ) -> None:
        self._model = model
        self._pose = pose
        self._root_frame = root_frame
        self._frames = frames
        self._frame_joints = frame_joints
        self._links = links
        self._shape = np.zeros(model.NUM_SHAPE_COEFFS, dtype=np.float32)
        self._identity = model.prepare_identity(self._shape, skip_vertices=True)

    @property
    def name(self) -> str:
        return self._root_frame.name

    @property
    def visible(self) -> bool:
        return self._root_frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._root_frame.visible = value

    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self._shape.copy()

    def set_shape(self, shape: Float[Array, "10"]) -> None:
        """Update the SMPL-X betas; bone lengths are re-solved, meshes stay."""
        if self._update_shape(shape):
            self._apply_pose()

    def get_pose(self) -> dict[str, Float[np.ndarray, "..."]]:
        """Return a copy of the currently applied pose parameters."""
        return {name: value.copy() for name, value in self._pose.items()}

    def set_pose(self, **params: Float[Array, "..."]) -> None:
        """Update any subset of the SMPL-X parameters and move the link meshes."""
        changed = False
        for name, value in params.items():
            if name == "expression":
                continue
            if name == "shape":
                changed |= self._update_shape(value)
                continue
            if name not in self._pose:
                raise KeyError(f"Unknown parameter {name!r}; expected one of {sorted(self._pose)}")
            value = np.asarray(value, dtype=self._pose[name].dtype)
            if value.shape != self._pose[name].shape:
                raise ValueError(
                    f"{name} must be unbatched with shape {self._pose[name].shape}, got {tuple(value.shape)}"
                )
            if np.array_equal(self._pose[name], value):
                continue
            self._pose[name] = value
            changed = True
        if changed:
            self._apply_pose()

    def remove(self) -> None:
        """Remove the mannequin from the scene."""
        for handle in (*self._links, *self._frames, self._root_frame):
            handle.remove()

    def _update_shape(self, shape: Float[Array, "10"]) -> bool:
        shape = np.asarray(shape, dtype=self._shape.dtype)
        if shape.shape != self._shape.shape:
            raise ValueError(f"shape must be unbatched with shape {self._shape.shape}, got {tuple(shape.shape)}")
        if np.array_equal(self._shape, shape):
            return False
        self._shape = shape
        self._identity = self._model.prepare_identity(shape, skip_vertices=True)
        return True

    def _apply_pose(self) -> None:
        skeleton = np.asarray(
            self._model.forward_skeleton(
                self._pose["body_pose"],
                self._pose["head_pose"],
                self._pose["hand_pose"],
                pelvis_rotation=self._pose["pelvis_rotation"],
                identity=self._identity,
                global_rotation=self._pose["global_rotation"],
                global_translation=self._pose["global_translation"],
            )
        )[self._frame_joints]
        wxyzs = SO3.conversions.from_rotmat_to_quat(skeleton[:, :3, :3], convention="wxyz", xp=np)
        positions = skeleton[:, :3, 3]
        for frame, wxyz, position in zip(self._frames, wxyzs, positions, strict=True):
            frame.wxyz = wxyz
            frame.position = position


def add_mannequin(
    scene: viser.SceneApi,
    name: str,
    model: SmplxMannequin,
    *,
    palette: str | tuple[Color, Color] = "sand",
    flat_shading: bool = True,
) -> ViserMannequinHandle:
    """Add a mannequin to a viser scene and return a handle for posing it.

    ``palette`` is either a name from :data:`PALETTES` or a custom
    ``(armor, joint)`` RGB pair.
    """
    if isinstance(palette, str):
        if palette not in PALETTES:
            raise ValueError(f"Unknown palette {palette!r}; available: {sorted(PALETTES)}")
        palette = PALETTES[palette]
    armor_color, joint_color = palette

    pose = model.get_rest_pose()
    del pose["shape"], pose["expression"]
    weights = model._weights

    root_frame = scene.add_frame(name, show_axes=False)
    frame_joints = sorted(set(weights.link_joint_indices))
    frames = [scene.add_frame(f"{name}/{weights.joint_names[joint]}", show_axes=False) for joint in frame_joints]

    links = []
    for link, (link_name, mesh) in enumerate(zip(model.link_names, model.link_meshes, strict=True)):
        part = link_name.split("__")[1]
        joint_name = weights.joint_names[weights.link_joint_indices[link]]
        links.append(
            scene.add_mesh_simple(
                f"{name}/{joint_name}/{part}",
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces),
                color=joint_color if "__joint_" in link_name else armor_color,
                flat_shading=flat_shading,
            )
        )
    handle = ViserMannequinHandle(model, pose, root_frame, frames, frame_joints, links)
    handle._apply_pose()
    return handle


__all__ = ["PALETTES", "ViserMannequinHandle", "add_mannequin"]
