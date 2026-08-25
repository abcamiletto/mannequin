"""Regenerate assets/smplx_calibration.npz from a body-models SMPL-X install.

Usage: python bake_calibration.py <SMPLX_NEUTRAL.npz> <output.npz>

Requires body-models (plus its SMPL-X model files) in the environment. SMPL-X
rest joints are exactly linear in the betas, so evaluating prepare_identity at
zero and at each unit beta captures the full shape response; the shaped-body
floor is recovered as the min over the (also linear) heights of the neutral
sole vertices (all vertices within 5cm of the neutral minimum).
"""

import sys

import numpy as np
from body_models.smplx.numpy import SMPLX

from mannequin import _io as io

MODEL_PATH, OUT = sys.argv[1:]
NUM_BETAS = 10
SOLE_BAND = 0.05

model = SMPLX(model_path=MODEL_PATH, flat_hand_mean=True)
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
rest_pose = model.get_rest_pose()
shape = rest_pose["shape"]
expression = rest_pose["expression"]
dominant_joint = np.asarray(model.skin_weights).argmax(axis=1)
regions = {
    "head": ("Head", "Jaw", "L_Eye", "R_Eye"),
}
region_masks = {
    name: np.isin(dominant_joint, [model.joint_names.index(joint) for joint in joints])
    for name, joints in regions.items()
}


def eval_shape(beta):
    identity = model.prepare_identity(np.asarray(beta, np.float64), expression)
    joints = np.asarray(identity["rest_joints"], np.float64)[index]
    vertices = np.asarray(identity["rest_vertices"], np.float64)
    bounds = {name: (vertices[mask].min(axis=0), vertices[mask].max(axis=0)) for name, mask in region_masks.items()}
    return joints, vertices, bounds


shape0 = np.zeros_like(shape)
joints0, vertices0, bounds0 = eval_shape(shape0)
y0 = vertices0[:, 1]
sole = np.where(y0 < y0.min() + SOLE_BAND)[0]

joint_dirs = np.zeros((len(index), 3, NUM_BETAS))
sole_dirs = np.zeros((len(sole), NUM_BETAS))
bound_dirs = {name: np.zeros((2, 3, NUM_BETAS)) for name in regions}
for i in range(NUM_BETAS):
    beta = shape0.copy()
    beta[i] = 1.0
    joints_i, vertices_i, bounds_i = eval_shape(beta)
    joint_dirs[:, :, i] = joints_i - joints0
    sole_dirs[:, i] = vertices_i[sole, 1] - y0[sole]
    for name in regions:
        bound_dirs[name][0, :, i] = bounds_i[name][0] - bounds0[name][0]
        bound_dirs[name][1, :, i] = bounds_i[name][1] - bounds0[name][1]

values = {
    "joint_rest": joints0.astype(np.float32),
    "joint_dirs": joint_dirs.astype(np.float32),
    "sole_y_rest": y0[sole].astype(np.float32),
    "sole_y_dirs": sole_dirs.astype(np.float32),
}
for name in regions:
    values[f"{name}_min_rest"] = bounds0[name][0].astype(np.float32)
    values[f"{name}_max_rest"] = bounds0[name][1].astype(np.float32)
    values[f"{name}_min_dirs"] = bound_dirs[name][0].astype(np.float32)
    values[f"{name}_max_dirs"] = bound_dirs[name][1].astype(np.float32)
np.savez_compressed(OUT, **values)
print("WROTE", OUT)
