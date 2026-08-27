"""Interactive pose-and-shape comparison of the bundled mannequins.

Run: uv run python examples/compare.py /path/to/SMPLX_NEUTRAL.npz
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser
from body_models.smplx.numpy import SMPLX

from mannequin import PALETTES, Mannequin, PaletteName, Pose, add_to_scene


class SmplxHandle:
    def __init__(self, scene: viser.SceneApi, model: Any, pose: Pose, palette: PaletteName) -> None:
        self._model = model
        self._pose = pose.copy()
        self._shape = np.zeros(model.NUM_SHAPE_COEFFS, dtype=np.float64)
        self._expression = np.zeros(model.NUM_EXPR_COEFFS, dtype=np.float64)
        self._head_pose = np.zeros((3, 3), dtype=np.float32)
        self._identity = model.prepare_identity(self._shape, self._expression)
        self._root = scene.add_frame("/smplx", show_axes=False)
        self._mesh = scene.add_mesh_simple(
            "/smplx/body",
            vertices=self._vertices(),
            faces=np.asarray(model.faces),
            color=PALETTES[palette][0],
        )

    def set_pose(self, pose: Pose) -> None:
        self._pose = pose.copy()
        self._mesh.vertices = self._vertices()

    def set_position(self, position: np.ndarray | tuple[float, float, float]) -> None:
        self._root.position = position

    def set_shape(self, shape: np.ndarray) -> None:
        self._shape.fill(0.0)
        self._shape[:10] = shape
        self._identity = self._model.prepare_identity(self._shape, self._expression)
        self._mesh.vertices = self._vertices()

    def set_palette(self, palette: PaletteName) -> None:
        self._mesh.color = PALETTES[palette][0]

    def _vertices(self) -> np.ndarray:
        return np.asarray(
            self._model.forward_vertices(
                self._pose.body,
                self._head_pose,
                self._pose.hands,
                identity=self._identity,
                global_rotation=self._pose.root_rotation,
                pelvis_rotation=self._pose.pelvis_rotation,
                global_translation=self._pose.translation,
            ),
            dtype=np.float32,
        )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("smplx_model", type=Path, help="Path to SMPLX_NEUTRAL.npz")
args = parser.parse_args()

server = viser.ViserServer(port=8080)
server.scene.set_up_direction("+y")
armor = Mannequin("armor", lod=1)
wooden = Mannequin("wooden")
smplx = SMPLX(model_path=args.smplx_model, flat_hand_mean=True)
armor_rest = armor.vertices(armor.rest_pose())
wooden_rest = wooden.vertices(wooden.rest_pose())
smplx_pose = armor.rest_pose()
smplx_rest = np.asarray(smplx.rest_vertices)
floor_y = float(min(armor_rest[:, 1].min(), wooden_rest[:, 1].min(), smplx_rest[:, 1].min()))
label_y = float(max(armor_rest[:, 1].max(), wooden_rest[:, 1].max(), smplx_rest[:, 1].max()) + 0.12)

server.scene.add_grid(
    "/floor",
    width=9.2,
    height=4.0,
    plane="xz",
    cell_size=0.1,
    section_size=0.5,
    position=(0.0, floor_y, 0.0),
)

armor_handle = add_to_scene(server.scene, "/armor", armor, palette="slate")
wooden_handle = add_to_scene(server.scene, "/wooden", wooden, palette="slate")
smplx_handle = SmplxHandle(server.scene, smplx, smplx_pose, "slate")
armor_label = server.scene.add_label(
    "/armor_label",
    "Armor",
    position=(-2.4, label_y, 0.0),
    anchor="bottom-center",
)
wooden_label = server.scene.add_label(
    "/wooden_label",
    "Wooden",
    position=(2.4, label_y, 0.0),
    anchor="bottom-center",
)
smplx_label = server.scene.add_label(
    "/smplx_label",
    "SMPL-X",
    position=(0.0, label_y, 0.0),
    anchor="bottom-center",
)

with server.gui.add_folder("View"):
    front_view_button = server.gui.add_button("Front view")
    side_view_button = server.gui.add_button("Side view")

with server.gui.add_folder("Pose"):
    playing = server.gui.add_checkbox("Animate", initial_value=True)
    phase_slider = server.gui.add_slider("Motion phase", min=0.0, max=1.0, step=0.005, initial_value=0.0)
    pose_strength = server.gui.add_slider("Pose strength", min=0.0, max=1.5, step=0.05, initial_value=1.0)
    speed_slider = server.gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=0.7)
    t_pose_button = server.gui.add_button("T pose")
    random_hands_button = server.gui.add_button("Random hand pose")

with server.gui.add_folder("Shape"):
    beta_sliders = tuple(
        server.gui.add_slider(f"Shape β{index}", min=-3.0, max=3.0, step=0.1, initial_value=0.0) for index in range(3)
    )

with server.gui.add_folder("Appearance"):
    palette_dropdown = server.gui.add_dropdown(
        "Color palette",
        options=tuple(PALETTES),
        initial_value="slate",
    )
    reset_shape = server.gui.add_button("Reset shape")
    server.gui.add_markdown(
        "The same SMPL-X shape coefficients drive all three figures. The reference "
        "uses the full body model; the mannequins map shape to bone lengths and rigid parts."
    )

FRONT_POSITIONS = ((-2.4, 0.0), (0.0, 0.0), (2.4, 0.0))
SIDE_POSITIONS = ((0.0, -2.4), (0.0, 0.0), (0.0, 2.4))
figures = (
    (armor_handle, armor_label),
    (smplx_handle, smplx_label),
    (wooden_handle, wooden_label),
)

updating_controls = False
random = np.random.default_rng()
hand_pose = np.zeros((30, 3), dtype=np.float32)
layout_positions = FRONT_POSITIONS


def apply_pose(phase: float) -> None:
    """Apply a looping asymmetric pose that exercises the major joints."""
    angle = phase * 2.0 * math.pi
    strength = float(pose_strength.value)
    swing = math.sin(angle) * strength
    bounce = math.sin(angle * 2.0) * strength
    pose = armor.rest_pose()
    pose.hands[:] = hand_pose
    body = pose.body
    body[0, 0] += 0.35 * swing
    body[1, 0] -= 0.35 * swing
    body[2, 2] += 0.08 * bounce
    body[12, 2] -= 0.12 * swing
    body[13, 2] -= 0.12 * swing
    body[15, 0] -= 0.5 * swing
    body[16, 0] += 0.5 * swing
    body[17, 2] += 0.55 * strength + 0.35 * swing
    body[18, 2] -= 0.55 * strength - 0.35 * swing
    body[6, 0] -= 0.18 * swing
    body[7, 0] += 0.18 * swing
    body[14, 1] += 0.18 * swing
    for (handle, _), (x, z) in zip(figures, layout_positions, strict=True):
        posed = pose.copy()
        posed.translation[1] = 0.025 * bounce
        handle.set_position((x, 0.0, z))
        handle.set_pose(posed)


def apply_shape() -> None:
    shape = np.zeros(10, dtype=np.float32)
    shape[:3] = [slider.value for slider in beta_sliders]
    with server.atomic():
        for handle, _ in figures:
            handle.set_shape(shape)
        apply_pose(float(phase_slider.value))


def set_t_pose() -> None:
    global updating_controls
    playing.value = False
    hand_pose.fill(0.0)
    updating_controls = True
    phase_slider.value = 0.0
    pose_strength.value = 0.0
    updating_controls = False
    with server.atomic():
        apply_pose(0.0)


def set_view(side: bool, client: viser.ClientHandle) -> None:
    global layout_positions
    layout_positions = SIDE_POSITIONS if side else FRONT_POSITIONS
    for (_, label), (x, z) in zip(figures, layout_positions, strict=True):
        label.position = (x, label_y, z)
    set_t_pose()
    client.camera.position = (8.0, 0.7, -1.3) if side else (0.5, 0.7, 8.0)
    client.camera.look_at = (0.0, floor_y + 0.85, -1.3) if side else (0.5, floor_y + 0.85, 0.0)
    client.camera.up_direction = (0.0, 1.0, 0.0)
    client.camera.fov = math.radians(35.0)


@front_view_button.on_click
def _(event: viser.GuiEvent) -> None:
    assert event.client is not None
    set_view(False, event.client)


@side_view_button.on_click
def _(event: viser.GuiEvent) -> None:
    assert event.client is not None
    set_view(True, event.client)


@phase_slider.on_update
def _(_) -> None:
    if not updating_controls:
        with server.atomic():
            apply_pose(float(phase_slider.value))


@pose_strength.on_update
def _(_) -> None:
    if not updating_controls:
        with server.atomic():
            apply_pose(float(phase_slider.value))


@t_pose_button.on_click
def _(_) -> None:
    set_t_pose()


@random_hands_button.on_click
def _(_) -> None:
    hand_pose[:] = np.clip(
        random.normal(0.0, 0.28, hand_pose.shape),
        -0.65,
        0.65,
    )
    playing.value = False
    with server.atomic():
        apply_pose(float(phase_slider.value))


for slider in beta_sliders:

    @slider.on_update
    def _(_) -> None:
        if not updating_controls:
            apply_shape()


@reset_shape.on_click
def _(_) -> None:
    global updating_controls
    updating_controls = True
    for slider in beta_sliders:
        slider.value = 0.0
    updating_controls = False
    apply_shape()


@palette_dropdown.on_update
def _(_) -> None:
    with server.atomic():
        for handle, _ in figures:
            handle.set_palette(palette_dropdown.value)


with server.atomic():
    apply_pose(float(phase_slider.value))
print(f"Open http://localhost:{server.get_port()} to compare the mannequins.")
last_tick = time.monotonic()
while True:
    now = time.monotonic()
    elapsed = now - last_tick
    last_tick = now
    if playing.value:
        phase = (float(phase_slider.value) + elapsed * float(speed_slider.value) * 0.35) % 1.0
        updating_controls = True
        phase_slider.value = phase
        updating_controls = False
        with server.atomic():
            apply_pose(phase)
    time.sleep(1.0 / 30.0)
