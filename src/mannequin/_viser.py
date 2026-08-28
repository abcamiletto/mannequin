"""Viser integration for :mod:`mannequin`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from nanomanifold import SO3

from mannequin._model import Kind, Mannequin, PoseParameters

if TYPE_CHECKING:
    import viser

Color = tuple[int, int, int]
PaletteName = Literal["sand", "ivory", "charcoal", "sage", "clay", "slate", "wood"]
Palette = PaletteName | tuple[Color, Color]
PALETTES: dict[PaletteName, tuple[Color, Color]] = {
    "sand": ((211, 151, 63), (133, 78, 20)),
    "ivory": ((245, 235, 210), (117, 91, 62)),
    "charcoal": ((42, 45, 53), (125, 138, 160)),
    "sage": ((126, 167, 111), (54, 96, 57)),
    "clay": ((211, 91, 57), (120, 48, 28)),
    "slate": ((77, 116, 161), (32, 59, 96)),
    "wood": ((181, 113, 54), (92, 52, 24)),
}
DEFAULT_PALETTES: dict[Kind, PaletteName] = {"armor": "sand", "wooden": "wood"}


@dataclass(frozen=True)
class _SkinMesh:
    handle: viser.MeshSkinnedHandle
    vertex_indices: np.ndarray
    joint_color: bool


class SceneHandle:
    """A live mannequin in a Viser scene."""

    def __init__(
        self,
        model: Mannequin,
        pose: PoseParameters,
        root: viser.FrameHandle,
        frames: list[viser.FrameHandle],
        frame_joints: list[int],
        meshes: list[viser.MeshHandle],
        skins: list[_SkinMesh],
        palette: Palette,
    ) -> None:
        self._model = model
        self._pose = _copy_pose(pose)
        self._root = root
        self._frames = frames
        self._frame_joints = frame_joints
        self._meshes = meshes
        self._skins = skins
        self._palette = palette

    @property
    def name(self) -> str:
        return self._root.name

    @property
    def visible(self) -> bool:
        return self._root.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._root.visible = value

    @property
    def pose(self) -> dict[str, np.ndarray]:
        return _copy_pose(self._pose)

    @property
    def shape(self) -> np.ndarray:
        return self._model.shape

    @property
    def palette(self) -> Palette:
        return self._palette

    @property
    def position(self) -> np.ndarray:
        return np.asarray(self._root.position).copy()

    def set_pose(self, pose: PoseParameters) -> None:
        """Set the mannequin pose."""
        self._pose = _copy_pose(pose)
        self._apply_pose()

    def set_position(self, position: np.ndarray | tuple[float, float, float]) -> None:
        """Set the mannequin's scene position."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"position must have shape (3,), got {position.shape}")
        self._root.position = position

    def set_shape(self, shape: np.ndarray) -> None:
        """Set the ten SMPL-X shape coefficients."""
        self._model.reshape(shape)
        self._pose["shape"] = self._model.shape
        self._update_shape()
        self._apply_pose()

    def set_palette(self, palette: Palette) -> None:
        """Set the body and joint colors."""
        self._palette = palette
        self._apply_palette()

    def remove(self) -> None:
        handles = [*self._meshes, *self._frames, *(skin.handle for skin in self._skins)]
        for handle in (*handles, self._root):
            handle.remove()

    def _update_shape(self) -> None:
        if self._skins:
            rest = self._model.rest_pose()
            _, positions = _skin_bones(self._model, rest)
            vertices = np.asarray(self._model.vertices(rest), dtype=np.float32)
            for skin in self._skins:
                skin.handle.vertices = vertices[skin.vertex_indices]
                skin.handle.bone_positions = positions
            return
        vertices = np.asarray(self._model._identity["link_local_vertices"], dtype=np.float32)
        for mesh, start, count in zip(
            self._meshes,
            self._model._weights.link_vertex_starts,
            self._model._weights.link_vertex_counts,
            strict=True,
        ):
            mesh.vertices = vertices[start : start + count]

    def _apply_pose(self) -> None:
        if self._skins:
            rotations, positions = _skin_bones(self._model, self._pose)
            for skin in self._skins:
                for bone, rotation, position in zip(skin.handle.bones, rotations, positions, strict=True):
                    bone.wxyz = rotation
                    bone.position = position
            return

        joints = np.asarray(self._model.joint_transforms(self._pose))[self._frame_joints]
        rotations = SO3.conversions.from_rotmat_to_quat(joints[:, :3, :3], convention="wxyz", xp=np)
        for frame, rotation, position in zip(self._frames, rotations, joints[:, :3, 3], strict=True):
            frame.wxyz = rotation
            frame.position = position

    def _apply_palette(self) -> None:
        body_color, joint_color = _palette_colors(self._palette)
        if self._skins:
            for skin in self._skins:
                skin.handle.color = joint_color if skin.joint_color else body_color
            return
        for mesh, link_name in zip(self._meshes, self._model.link_names, strict=True):
            mesh.color = joint_color if "__joint_" in link_name else body_color


def add_to_scene(
    scene: viser.SceneApi,
    name: str,
    model: Mannequin,
    *,
    palette: Palette | None = None,
    flat_shading: bool | None = None,
) -> SceneHandle:
    """Add a mannequin to a Viser scene."""
    palette = DEFAULT_PALETTES[model.kind] if palette is None else palette
    flat_shading = not model.skinned if flat_shading is None else flat_shading
    body_color, joint_color = _palette_colors(palette)
    pose = model.rest_pose()
    root = scene.add_frame(name, show_axes=False)

    if model.skinned:
        skins = _add_skinned_meshes(scene, name, model, pose, body_color, joint_color, flat_shading)
        handle = SceneHandle(model, pose, root, [], [], [], skins, palette)
    else:
        frames, frame_joints, meshes = _add_rigid_meshes(
            scene,
            name,
            model,
            body_color,
            joint_color,
            flat_shading,
        )
        handle = SceneHandle(model, pose, root, frames, frame_joints, meshes, [], palette)
    handle._apply_pose()
    return handle


def _palette_colors(palette: Palette) -> tuple[Color, Color]:
    return PALETTES[palette] if isinstance(palette, str) else palette


def _copy_pose(pose: PoseParameters) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in pose.items()}


def _add_skinned_meshes(
    scene: viser.SceneApi,
    name: str,
    model: Mannequin,
    pose: PoseParameters,
    body_color: Color,
    joint_color: Color,
    flat_shading: bool,
) -> list[_SkinMesh]:
    weights = model._weights
    vertices = np.asarray(model.vertices(pose), dtype=np.float32)
    joint_indices = model._skin_joint_indices
    skin_weights = model._skin_weights
    assert joint_indices is not None and skin_weights is not None
    dense_weights = np.zeros((model.num_vertices, len(model.joint_names)), dtype=np.float32)
    rows = np.arange(model.num_vertices)[:, None]
    np.add.at(dense_weights, (rows, joint_indices), skin_weights)
    rotations, positions = _skin_bones(model, pose)
    part_names = weights.skin_part_names
    vertex_starts = weights.skin_part_vertex_starts
    vertex_counts = weights.skin_part_vertex_counts
    face_starts = weights.skin_part_face_starts
    face_counts = weights.skin_part_face_counts
    assert part_names is not None
    assert vertex_starts is not None and vertex_counts is not None
    assert face_starts is not None and face_counts is not None

    groups = {
        "body": [index for index, part in enumerate(part_names) if not part.startswith("joint_")],
        "joints": [index for index, part in enumerate(part_names) if part.startswith("joint_")],
    }
    skins = []
    for group_name, parts in groups.items():
        if not parts:
            continue
        vertex_indices = np.concatenate(
            [np.arange(vertex_starts[index], vertex_starts[index] + vertex_counts[index]) for index in parts]
        )
        remap = np.full(model.num_vertices, -1, dtype=np.int64)
        remap[vertex_indices] = np.arange(len(vertex_indices))
        faces = np.concatenate(
            [weights.faces[face_starts[index] : face_starts[index] + face_counts[index]] for index in parts]
        )
        joint_group = group_name == "joints"
        skin = scene.add_mesh_skinned(
            f"{name}/{group_name}",
            vertices[vertex_indices],
            remap[faces],
            bone_wxyzs=rotations,
            bone_positions=positions,
            skin_weights=dense_weights[vertex_indices],
            color=joint_color if joint_group else body_color,
            flat_shading=flat_shading,
        )
        skins.append(_SkinMesh(skin, vertex_indices, joint_group))
    return skins


def _skin_bones(model: Mannequin, pose: PoseParameters) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(model.joint_transforms(pose))
    rotations = SO3.conversions.from_rotmat_to_quat(joints[:, :3, :3], convention="wxyz", xp=np)
    return rotations, joints[:, :3, 3]


def _add_rigid_meshes(
    scene: viser.SceneApi,
    name: str,
    model: Mannequin,
    body_color: Color,
    joint_color: Color,
    flat_shading: bool,
) -> tuple[list[viser.FrameHandle], list[int], list[viser.MeshHandle]]:
    weights = model._weights
    frame_joints = sorted(set(weights.link_joint_indices))
    frames = [scene.add_frame(f"{name}/{weights.joint_names[joint]}", show_axes=False) for joint in frame_joints]
    meshes = []
    vertices = model._identity["link_local_vertices"]
    for link_name, joint, vertex_start, vertex_count, face_start, face_count in zip(
        model.link_names,
        weights.link_joint_indices,
        weights.link_vertex_starts,
        weights.link_vertex_counts,
        weights.link_face_starts,
        weights.link_face_counts,
        strict=True,
    ):
        part = link_name.split("__")[1]
        faces = weights.faces[face_start : face_start + face_count] - vertex_start
        meshes.append(
            scene.add_mesh_simple(
                f"{name}/{weights.joint_names[joint]}/{part}",
                vertices=np.asarray(vertices[vertex_start : vertex_start + vertex_count], dtype=np.float32),
                faces=np.asarray(faces),
                color=joint_color if "__joint_" in link_name else body_color,
                flat_shading=flat_shading,
            )
        )
    return frames, frame_joints, meshes


__all__ = ["PALETTES", "Palette", "PaletteName", "SceneHandle", "add_to_scene"]
