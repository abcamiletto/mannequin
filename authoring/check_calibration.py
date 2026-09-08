"""Measure a mannequin against a local neutral SMPL-X model.

uv run python authoring/check_calibration.py SMPLX_NEUTRAL.npz atelier report.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from body_models.smplx.numpy import SMPLX

from mannequin import Mannequin


def main():
    model_path, kind, output = sys.argv[1:]
    reference = SMPLX(model_path=model_path, flat_hand_mean=True)
    model = Mannequin(kind)
    aliases = {"L_Hand": "L_Wrist", "R_Hand": "R_Wrist"}
    mapping = [reference.joint_names.index(aliases.get(name, name)) for name in model.joint_names]
    random = np.random.default_rng(20260908)
    shapes = np.concatenate((np.zeros((1, 10)), 2 * np.eye(10), -2 * np.eye(10), random.uniform(-2, 2, (4, 10))))
    max_position = max_rotation = max_floor = max_height = 0.0
    for shape in shapes:
        model.reshape(shape)
        neutral = reference.get_rest_pose()
        neutral["shape"][:10] = shape
        ref_vertices = reference.forward_vertices(**neutral)
        vertices = model.vertices(model.rest_pose())
        max_floor = max(max_floor, abs(vertices[:, 1].min() - ref_vertices[:, 1].min()))
        max_height = max(max_height, abs(np.ptp(vertices[:, 1]) - np.ptp(ref_vertices[:, 1])))
        for pose_index in range(7):
            pose = model.rest_pose()
            if pose_index:
                pose["body_pose"] = random.uniform(-0.8, 0.8, (21, 3))
                pose["hand_pose"] = random.uniform(-0.5, 0.5, (30, 3))
                pose["pelvis_rotation"] = random.uniform(-0.4, 0.4, 3)
                pose["global_rotation"] = random.uniform(-0.4, 0.4, 3)
                pose["global_translation"] = random.uniform(-1, 1, 3)
            ref_pose = {**pose, "shape": neutral["shape"]}
            expected = reference.forward_skeleton(**ref_pose)[mapping]
            actual = model.joint_transforms(pose)
            position_error = np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1).max()
            max_position = max(max_position, position_error)
            max_rotation = max(max_rotation, np.abs(actual[:, :3, :3] - expected[:, :3, :3]).max())
    report = {
        "kind": kind,
        "reference": "SMPLX_NEUTRAL.npz",
        "seed": 20260908,
        "shape_cases": len(shapes),
        "pose_cases": len(shapes) * 7,
        "mapped_joints": len(mapping),
        "max_joint_position_error_m": float(max_position),
        "max_joint_rotation_matrix_error": float(max_rotation),
        "max_rest_floor_error_m": float(max_floor),
        "max_rest_height_error_m": float(max_height),
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if max_position > 1e-5 or max_rotation > 1e-5 or max_floor > 1e-5 or max_height > 0.005:
        raise RuntimeError("Calibration exceeds joint/floor (10 µm) or height (5 mm) tolerance")


if __name__ == "__main__":
    main()
