# SMPL-X Mannequin

`mannequin-x` provides two lightweight figures driven by SMPL-X body and hand
rotations:

- `armor` is the repo's segmented rigid mannequin, available in three LODs.
- `wooden` is a skinned wooden mannequin at its source resolution.

Both designs accept the same `Pose`. Ten SMPL-X shape coefficients resize their
bones and geometry. The NumPy runtime includes the required shape calibration,
so it does not need SMPL-X model files.

```bash
pip install mannequin-x
```

## Python API

```python
import numpy as np

from mannequin import Mannequin

shape = np.zeros(10, dtype=np.float32)
shape[0] = 1.5

model = Mannequin("wooden", shape=shape)
pose = model.rest_pose()
pose.body[17, 2] = 0.8

vertices = model.vertices(pose)
faces = model.faces
joint_transforms = model.joint_transforms(pose)
```

Create armor with `Mannequin("armor", lod=0)`. Armor supports LODs 0, 1, and
2. The wooden model has one resolution, so it does not accept `lod`.

`rest_pose()` returns a mutable `Pose` with four arrays:

- `body`: `[..., 21, 3]` SMPL-X body rotations
- `hands`: `[..., 30, 3]` left and right hand rotations
- `root_rotation`: `[..., 3]` world rotation
- `translation`: `[..., 3]` world translation

Rotations use axis-angle vectors. Leading batch dimensions are supported.
Jaw, eye, and expression parameters are absent because neither mannequin has
matching geometry or joints.

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

The viewer shows armor, wooden, and full SMPL-X figures with matched motion,
random hand poses, shape controls, front and side views, a T-pose button, and a
shared palette. The SMPL-X file remains external to the package.

## three.js

`authoring/export_glb.py` exports rigid armor as a nested GLB joint hierarchy.
The root extras contain the SMPL-X body and hand parameter mapping. See
[`examples/threejs.html`](examples/threejs.html).

## Assets

The editable armor source is [`authoring/mannequin.blend`](authoring/mannequin.blend).

`src/mannequin/assets/wooden.npz` retains its source vertices and skin weights,
converts coordinates, triangulates faces, and maps 52 source bones onto this
package's joint hierarchy. Rebuild it with `authoring/import_wooden.py`.

`authoring/bake_calibration.py` uses the NumPy SMPL-X implementation from
`body-models` to bake the joint, ground-plane, head, and height response for the
first ten shape coefficients. Pass it a local `SMPLX_NEUTRAL.npz`; the runtime
does not depend on `body-models` or the SMPL-X file. The wooden torso keeps its
source proportions.
