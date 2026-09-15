# SMPL-X Mannequin

`mannequin-x` provides two lightweight figures driven by SMPL-X body and hand
rotations. `convex` is a simulation-friendly human made from rigid convex
hulls in three LODs. `wooden`, the default, is a skinned mannequin at its
source resolution.

All designs accept pose dictionaries with the same fields as `body-models`.
Ten SMPL-X shape coefficients resize their bones and geometry. The NumPy
runtime includes the required shape calibration, so it does not need SMPL-X
model files.

```bash
pip install mannequin-x
```

## Python API

```python
import numpy as np

from mannequin import Mannequin

shape = np.zeros(10, dtype=np.float32)
shape[0] = 1.5

model = Mannequin("wooden", shape=shape, flat_hand_mean=False)
pose = model.rest_pose()
pose["body_pose"][17, 2] = 0.8

vertices = model.vertices(pose)
faces = model.faces
joint_transforms = model.joint_transforms(pose)
```

The same methods accept the parameter dictionary returned by `body-models`
without renaming fields:

```python
from body_models.smplx.numpy import SMPLX

smplx = SMPLX(model_path="SMPLX_NEUTRAL.npz", flat_hand_mean=False)
rest = smplx.get_rest_pose()

joint_transforms = model.forward_skeleton(**rest)
# Equivalent shorthand:
joint_transforms = model.joint_transforms(rest)
vertices = model.vertices(rest)
```

`flat_hand_mean` defaults to `False`, matching SMPL-X. In this mode, zero hand
parameters produce the relaxed mean hand pose. Pass `flat_hand_mean=True` when
using SMPL-X parameters created with the flat-hand convention.

`forward_skeleton()` accepts the `body_pose`, `head_pose`, `hand_pose`,
`pelvis_rotation`, `shape`, `expression`, `global_rotation`, and
`global_translation` fields from `body-models`. `rest_pose()` returns those
same fields. `joint_names` uses the SMPL-X names `Spine1`, `Spine2`, `Spine3`,
`L_Foot`, `R_Foot`, `L_Collar`, and `R_Collar`. The mannequin omits the jaw and
eye joints and adds zero-length `L_Hand` and `R_Hand` skinning joints at the
wrists. It accepts but ignores `head_pose` and `expression` because neither
mannequin has the corresponding joints or geometry.

Create the convex model with `Mannequin("convex", lod=0)`. Its LODs contain 80,
52, and 22 hulls respectively. The wooden model has one resolution, so it does
not accept `lod`.

`rest_pose()` returns a mutable dictionary with the `body-models` fields:

- `body_pose`: `[..., 21, 3]` SMPL-X body rotations
- `head_pose`: `[..., 3, 3]` unused jaw and eye rotations
- `hand_pose`: `[..., 30, 3]` left and right hand rotations
- `global_rotation`: `[..., 3]` world rotation
- `pelvis_rotation`: `[..., 3]` pelvis rotation about the pelvis joint
- `global_translation`: `[..., 3]` world translation
- `shape`: `[..., 10]` mannequin shape coefficients
- `expression`: `[..., 10]` unused expression coefficients

`global_rotation` rotates the whole figure around the SMPL-X origin.
`pelvis_rotation` rotates the body around the pelvis without moving the pelvis.
Rotations use axis-angle vectors. Leading batch dimensions are supported.

Shape is identity state, not motion state. Set it at construction or call
`model.reshape(shape)`. Pose evaluation then stays concise:

```python
model.reshape(new_shape)
vertices = model.vertices(pose)
links = model.link_transforms(pose)
```

## Viser

Install the optional viewer dependency with `pip install mannequin-x[viser]`.

```python
import viser

from mannequin import Mannequin, add_to_scene

server = viser.ViserServer()
model = Mannequin("wooden")
handle = add_to_scene(server.scene, "/mannequin", model)

handle.set_pose(model.rest_pose())
handle.set_position((1.0, 0.0, 0.0))
handle.set_shape(new_shape)
handle.set_palette("sage")
```

Pass one of `sand`, `ivory`, `charcoal`, `sage`, `clay`, `slate`, or `wood` to
`add_to_scene(..., palette=...)` or `handle.set_palette(...)`.
Wooden meshes use Viser's native `add_mesh_skinned()`, so pose updates send
bone transforms instead of vertex buffers.

Run the live comparison against a local neutral SMPL-X model:

```bash
uv run python examples/compare.py /path/to/SMPLX_NEUTRAL.npz
```

The viewer shows convex, wooden, and full SMPL-X figures with matched motion,
random hand poses, shape controls, front and side views, a T-pose button, and a
shared palette. The SMPL-X file remains external to the package.

## Assets

`src/mannequin/assets/wooden.npz` retains its source vertices and skin weights,
converts coordinates, triangulates faces, and maps 52 source bones onto this
package's joint hierarchy. Rebuild it with `authoring/import_wooden.py`.

`authoring/bake_calibration.py` uses the NumPy SMPL-X implementation from
`body-models` to bake the joint, ground-plane, head, and height response for the
first ten shape coefficients. Pass it a local `SMPLX_NEUTRAL.npz`; the runtime
does not depend on `body-models` or the SMPL-X file. The wooden torso keeps its
source proportions.
