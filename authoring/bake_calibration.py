"""Regenerate assets/smplx_calibration.npz from a body-models SMPL-X install.

Usage: python bake_calibration.py <output.npz>

Requires body-models (plus its SMPL-X model files) in the environment. SMPL-X
rest joints are exactly linear in the betas, so evaluating prepare_identity at
zero and at each unit beta captures the full shape response; the shaped-body
floor is recovered as the min over the (also linear) heights of the neutral
sole vertices (all vertices within 5cm of the neutral minimum).
"""

import sys

import numpy as np
from body_models.smplx import SMPLX
from body_models.smplx._constants import SMPLX_BODY_PRESETS, SMPLX_HAND_PRESETS

from mannequin import _io as io

OUT = sys.argv[1]
NUM_BETAS = 10
SOLE_BAND = 0.05

model = SMPLX(gender="neutral", flat_hand_mean=True, runtime="numpy")
weights = io.load(0)
renamed = {
    "Torso": "Spine1",
    "Spine": "Spine2",
    "Chest": "Spine3",
    "L_Toe": "L_Foot",
    "R_Toe": "R_Foot",
    "L_Thorax": "L_Collar",
    "R_Thorax": "R_Collar",
    "L_Hand": "L_Wrist",
    "R_Hand": "R_Wrist",
}
index = [model.joint_names.index(renamed.get(name, name)) for name in weights.joint_names]
zeros = np.zeros(NUM_BETAS, dtype=np.float64)


def eval_shape(beta):
    ident = model.prepare_identity(np.asarray(beta, np.float64), zeros)
    return np.asarray(ident["rest_joints"], np.float64)[index], np.asarray(ident["rest_vertices"], np.float64)


joints0, vertices0 = eval_shape(zeros)
y0 = vertices0[:, 1]
sole = np.where(y0 < y0.min() + SOLE_BAND)[0]

joint_dirs = np.zeros((len(index), 3, NUM_BETAS))
sole_dirs = np.zeros((len(sole), NUM_BETAS))
for i in range(NUM_BETAS):
    beta = zeros.copy()
    beta[i] = 1.0
    joints_i, vertices_i = eval_shape(beta)
    joint_dirs[:, :, i] = joints_i - joints0
    sole_dirs[:, i] = vertices_i[sole, 1] - y0[sole]

np.savez_compressed(
    OUT,
    joint_rest=joints0.astype(np.float32),
    joint_dirs=joint_dirs.astype(np.float32),
    sole_y_rest=y0[sole].astype(np.float32),
    sole_y_dirs=sole_dirs.astype(np.float32),
    body_a_pose=np.asarray(SMPLX_BODY_PRESETS["a_pose"], np.float32),
    hand_flat=np.asarray(SMPLX_HAND_PRESETS["flat"], np.float32).reshape(30, 3),
    hand_rest=np.asarray(SMPLX_HAND_PRESETS["rest"], np.float32).reshape(30, 3),
)
print("WROTE", OUT)
