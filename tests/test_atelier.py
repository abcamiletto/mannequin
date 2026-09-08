"""Regressions for the Atelier shoulder attachments and packaged skin binding."""

import hashlib
import unittest

import numpy as np

from mannequin import Mannequin, _io


class AtelierTests(unittest.TestCase):
    def test_binding_and_mesh(self):
        weights = _io.load(0, kind="atelier")
        assert weights.skin_weights is not None
        self.assertEqual(weights.skin_weights.shape, (len(weights.vertices), 4))
        self.assertTrue(np.all(weights.skin_weights >= 0))
        np.testing.assert_allclose(weights.skin_weights.sum(axis=1), 1, atol=1e-7)
        self.assertTrue(np.all(np.isfinite(weights.vertices)))
        self.assertGreaterEqual(weights.faces.min(), 0)
        self.assertLess(weights.faces.max(), len(weights.vertices))
        # Every shell must remain closed after replacing the upper-arm meshes.
        edges = np.sort(
            np.concatenate([weights.faces[:, [0, 1]], weights.faces[:, [1, 2]], weights.faces[:, [2, 0]]]), axis=1
        )
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)

    def test_original_head(self):
        weights = _io.load(0, kind="atelier")
        assert weights.skin_part_names is not None
        assert weights.skin_part_vertex_starts is not None
        assert weights.skin_part_vertex_counts is not None
        part = weights.skin_part_names.index("head")
        start, count = weights.skin_part_vertex_starts[part], weights.skin_part_vertex_counts[part]
        digest = hashlib.sha256(weights.vertices[start : start + count].astype(np.float32).tobytes()).hexdigest()
        self.assertEqual(digest, "a3352792690edce4de7d9d19e9302bfeff48fbb60356180b4327d377a5c2b185")

    def test_socket_rims_follow_shoulders(self):
        # Full collar-weight socket vertices must stay the same distance from
        # the shoulder pivot during clavicle motion and independent arm rotation.
        for size in (-2.0, 0.0, 2.0):
            shape = np.zeros(10)
            shape[0] = size
            model = Mannequin("atelier", shape=shape)
            data = model._weights
            assert data.skin_part_names is not None
            assert data.skin_part_vertex_starts is not None
            assert data.skin_part_vertex_counts is not None
            assert data.skin_joint_indices is not None
            assert data.skin_weights is not None
            body = data.skin_part_names.index("body")
            start, count = data.skin_part_vertex_starts[body], data.skin_part_vertex_counts[body]
            rest = model.rest_pose()
            rest_vertices, rest_joints = model.vertices(rest), model.joint_transforms(rest)
            for side, collar_pose, arm_pose in (("L", 12, 15), ("R", 13, 16)):
                collar = data.joint_names.index(f"{side}_Thorax")
                shoulder = data.joint_names.index(f"{side}_Shoulder")
                ids = np.arange(start, start + count)
                influence = np.sum(np.where(data.skin_joint_indices[ids] == collar, data.skin_weights[ids], 0), axis=1)
                ids = ids[influence > 1 - 1e-7]
                self.assertGreater(len(ids), 10)
                rest_radius = np.linalg.norm(rest_vertices[ids] - rest_joints[shoulder, :3, 3], axis=1)
                for axis in range(3):
                    for angle in np.linspace(-np.pi, np.pi, 9):
                        pose = model.rest_pose()
                        pose["body_pose"][collar_pose, axis] = 0.4 * np.sin(angle)
                        pose["body_pose"][arm_pose, axis] = angle
                        vertices, joints = model.vertices(pose), model.joint_transforms(pose)
                        self.assertTrue(np.isfinite(vertices).all())
                        radii = np.linalg.norm(vertices[ids] - joints[shoulder, :3, 3], axis=1)
                        np.testing.assert_allclose(radii, rest_radius, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
