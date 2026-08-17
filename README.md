# SMPL-X Mannequin

A rigid mannequin driven directly by SMPL-X parameters — no skinning, no
blend shapes. Every part, down to the fingers, is a rigid mesh owned by one
SMPL-X joint; shape coefficients change bone lengths without changing part
thickness. Numpy-only and fully standalone: the SMPL-X shape response is
baked into a small bundled table, so neither `body-models` nor the SMPL-X
model files are needed.

```bash
pip install mannequin-x
```

## Usage

```python
from mannequin import SmplxMannequin

model = SmplxMannequin(lod=1)
params = model.get_rest_pose()
vertices = model.forward_vertices(**params)
```

`forward_vertices`, `forward_skeleton`, `forward_links`, and `forward_meshes`
take SMPL-X parameters (axis-angle, optionally batched) and use the native
SMPL-X origin: identical parameters place the pelvis and neutral foot surface
in the same coordinates as SMPL-X. For repeated calls with the same betas,
prepare the identity once with `prepare_identity` and pass it as `identity=`.

## Viser

Each link mesh is uploaded to the [viser](https://viser.studio) scene once,
parented to its joint's frame; pose and shape updates only send the per-joint
frame transforms (~10 KB per full pose).

```python
import viser
from mannequin import SmplxMannequin, add_mannequin

server = viser.ViserServer()
model = SmplxMannequin(lod=1)
handle = add_mannequin(server.scene, "/mannequin", model)
handle.set_shape(betas)
handle.set_pose(**model.get_apose())
```

`set_pose` accepts any subset of the SMPL-X pose parameters, and full
per-frame parameter dicts work directly (`shape` is routed to `set_shape`,
`expression` is ignored). Unchanged values are skipped.

`add_mannequin` takes a `palette` — one of the built-in armor/joint pairs in
`PALETTES` (`sand`, `ivory`, `charcoal`, `sage`, `clay`, `slate`; see
[`renders/palettes.jpg`](renders/palettes.jpg)) or a custom
`(armor, joint)` RGB pair.

## Assets

Three exactly mirrored levels of detail are bundled: `lod=0` (~40k vertices),
`lod=1` (~15k), and `lod=2` (<5k). Lower-leg length is calibrated against the
SMPL-X sole vertices, so shaped identities keep the same ground plane. The
editable source is [`authoring/mannequin.blend`](authoring/mannequin.blend);
the installed package ships only the compact numpy assets.
