"""Rigid and skinned mannequins driven by SMPL-X motion."""

from mannequin._model import Kind, Mannequin, Pose
from mannequin._viser import PALETTES, Palette, PaletteName, SceneHandle, add_to_scene

__all__ = ["PALETTES", "Kind", "Mannequin", "Palette", "PaletteName", "Pose", "SceneHandle", "add_to_scene"]
