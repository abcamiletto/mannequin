"""Rigid, non-skinned mannequin driven directly by SMPL-X motion."""

from mannequin._model import SmplxMannequin
from mannequin._viser import PALETTES, ViserMannequinHandle, add_mannequin

__all__ = ["PALETTES", "SmplxMannequin", "ViserMannequinHandle", "add_mannequin"]
