"""Regression test against outputs recorded from the body-models-backed implementation."""

from pathlib import Path

import numpy as np
import pytest

from mannequin import SmplxMannequin

GOLDEN = np.load(Path(__file__).parent / "golden.npz")


@pytest.mark.parametrize("lod", [0, 2])
def test_rest_and_apose_match_golden(lod):
    model = SmplxMannequin(lod=lod)
    rest = model.get_rest_pose()
    np.testing.assert_allclose(model.forward_vertices(**rest), GOLDEN[f"lod{lod}_rest_verts"], atol=1e-5, rtol=0)
    apose = model.get_apose(hands="rest")
    np.testing.assert_allclose(apose["body_pose"], GOLDEN[f"lod{lod}_apose_body"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(apose["hand_pose"], GOLDEN[f"lod{lod}_apose_hands"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(model.forward_vertices(**apose), GOLDEN[f"lod{lod}_apose_verts"], atol=1e-5, rtol=0)


@pytest.mark.parametrize("cfg", [0, 1, 2])
def test_shaped_posed_outputs_match_golden(cfg):
    model = SmplxMannequin(lod=2)
    shape = GOLDEN[f"cfg{cfg}_shape"]
    body = GOLDEN[f"cfg{cfg}_body"]
    hands = GOLDEN[f"cfg{cfg}_hands"]
    head = np.zeros((3, 3), np.float32)
    kwargs = {
        "pelvis_rotation": GOLDEN[f"cfg{cfg}_pelvis"],
        "global_rotation": GOLDEN[f"cfg{cfg}_grot"],
        "global_translation": GOLDEN[f"cfg{cfg}_gtra"],
    }
    identity = model.prepare_identity(shape)
    if cfg == 0:
        np.testing.assert_allclose(identity["local_joint_offsets"], GOLDEN["cfg0_local_offsets"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(identity["rest_joints"], GOLDEN["cfg0_rest_joints"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        model.forward_vertices(body, head, hands, identity=identity, **kwargs),
        GOLDEN[f"cfg{cfg}_verts"],
        atol=2e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        model.forward_skeleton(body, head, hands, identity=identity, **kwargs),
        GOLDEN[f"cfg{cfg}_skel"],
        atol=2e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        model.forward_links(body, head, hands, identity=identity, **kwargs),
        GOLDEN[f"cfg{cfg}_links"],
        atol=2e-5,
        rtol=0,
    )


def test_batched_forward_matches_golden():
    model = SmplxMannequin(lod=2)
    body = GOLDEN["batch_body"]
    hands = np.zeros((2, 30, 3), np.float32)
    head = np.zeros((2, 3, 3), np.float32)
    np.testing.assert_allclose(model.forward_vertices(body, head, hands), GOLDEN["batch_verts"], atol=1e-5, rtol=0)
