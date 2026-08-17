"""Tests for the viser handle using a duck-typed stub scene (no viser required)."""

import numpy as np
import pytest
from nanomanifold import SO3

from mannequin import SmplxMannequin, add_mannequin


class StubHandle:
    def __init__(self, name, wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0)):
        self.name = name
        self.wxyz = np.asarray(wxyz, np.float64)
        self.position = np.asarray(position, np.float64)
        self.visible = True
        self.updates = 0

    def __setattr__(self, key, value):
        if key in ("wxyz", "position") and "updates" in self.__dict__:
            self.__dict__["updates"] += 1
        self.__dict__[key] = value

    def remove(self):
        pass


class StubScene:
    def __init__(self):
        self.mesh_adds = 0

    def add_frame(self, name, show_axes=False):
        return StubHandle(name)

    def add_mesh_simple(self, name, vertices, faces, color, flat_shading, wxyz, position):
        self.mesh_adds += 1
        return StubHandle(name, wxyz=wxyz, position=position)


@pytest.fixture(scope="module")
def model():
    return SmplxMannequin(lod=2)


def assert_links_match(handle, model, **forward_kwargs):
    """Frame transform composed with the static mesh offset must equal forward_links."""
    expected = np.asarray(model.forward_links(**forward_kwargs))
    frame_by_joint = dict(zip(handle._frame_joints, handle._frames, strict=True))
    for index, link in enumerate(handle._links):
        frame = frame_by_joint[model._weights.link_joint_indices[index]]
        frame_rot = SO3.conversions.from_quat_to_rotmat(np.asarray(frame.wxyz), convention="wxyz", xp=np)
        link_rot = SO3.conversions.from_quat_to_rotmat(np.asarray(link.wxyz), convention="wxyz", xp=np)
        rotation = frame_rot @ link_rot
        position = np.asarray(frame.position) + frame_rot @ np.asarray(link.position)
        np.testing.assert_allclose(rotation, expected[index, :3, :3], atol=1e-5, rtol=0)
        np.testing.assert_allclose(position, expected[index, :3, 3], atol=1e-5, rtol=0)


def test_meshes_added_once_and_pose_updates_move_frames(model):
    scene = StubScene()
    handle = add_mannequin(scene, "/mannequin", model)
    assert scene.mesh_adds == len(model.link_names)

    apose = model.get_apose(hands="rest")
    handle.set_pose(**apose)
    assert scene.mesh_adds == len(model.link_names)
    assert all(link.updates == 0 for link in handle._links)
    assert_links_match(
        handle,
        model,
        body_pose=apose["body_pose"],
        head_pose=apose["head_pose"],
        hand_pose=apose["hand_pose"],
        pelvis_rotation=apose["pelvis_rotation"],
        global_rotation=apose["global_rotation"],
        global_translation=apose["global_translation"],
    )


def test_unchanged_pose_skips_recompute(model):
    scene = StubScene()
    handle = add_mannequin(scene, "/mannequin", model)
    apose = model.get_apose()
    handle.set_pose(**apose)
    before = [frame.updates for frame in handle._frames]
    handle.set_pose(**apose)
    assert [frame.updates for frame in handle._frames] == before


def test_set_shape_moves_frames_without_readding_meshes(model):
    scene = StubScene()
    handle = add_mannequin(scene, "/mannequin", model)
    tall = np.zeros(10, np.float32)
    tall[0] = 2.0
    positions = [frame.position.copy() for frame in handle._frames]
    handle.set_shape(tall)
    assert scene.mesh_adds == len(model.link_names)
    assert all(link.updates == 0 for link in handle._links)
    assert any(not np.allclose(frame.position, old) for frame, old in zip(handle._frames, positions, strict=True))
    np.testing.assert_array_equal(handle.shape, tall)
    identity = model.prepare_identity(tall)
    rest = model.get_rest_pose()
    assert_links_match(
        handle,
        model,
        body_pose=rest["body_pose"],
        head_pose=rest["head_pose"],
        hand_pose=rest["hand_pose"],
        pelvis_rotation=rest["pelvis_rotation"],
        identity=identity,
        global_rotation=rest["global_rotation"],
        global_translation=rest["global_translation"],
    )


def test_set_pose_routes_shape_and_ignores_expression(model):
    scene = StubScene()
    handle = add_mannequin(scene, "/mannequin", model)
    tall = np.zeros(10, np.float32)
    tall[0] = 2.0
    handle.set_pose(shape=tall, expression=np.ones(10, np.float32))
    np.testing.assert_array_equal(handle.shape, tall)
    assert "shape" not in handle.get_pose()


def test_palette_selection(model):
    add_mannequin(StubScene(), "/mannequin", model, palette="charcoal")
    add_mannequin(StubScene(), "/mannequin", model, palette=((10, 20, 30), (40, 50, 60)))
    with pytest.raises(ValueError, match="Unknown palette"):
        add_mannequin(StubScene(), "/mannequin", model, palette="neon")


def test_invalid_parameters_are_rejected(model):
    scene = StubScene()
    handle = add_mannequin(scene, "/mannequin", model)
    with pytest.raises(KeyError):
        handle.set_pose(qpos=np.zeros(3))
    with pytest.raises(ValueError):
        handle.set_pose(body_pose=np.zeros((2, 21, 3), np.float32))
    with pytest.raises(ValueError):
        handle.set_shape(np.zeros((2, 10), np.float32))
