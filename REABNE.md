# STAG4D Motion Only

This is a clean motion-only project extracted for testing dynamic reconstruction from:

- an already good canonical 3D Gaussian model
- a front-view video sequence of the same object

It does not run canonical generation. It only trains the dynamic model:

- `HexPlane/Grid`
- `VelocityField + VideoEncoder + ODE`

The original deformation-point selection logic is kept through `deformation_table`.

## Inputs

Canonical model:

```text
canonical_dir/
  model.ply
  optional: deformation.pth deformation_table.pth deformation_accum.pth
```

Front-view video frames:

```text
video_path/
  000_rgba.png
  001_rgba.png
  002_rgba.png
  ...
```

or RGB frames with separate masks:

```text
video_path/
  000.png
  001.png
  ...
mask_path/
  000.png
  001.png
  ...
```

## Training behavior

- skips canonical reconstruction entirely
- uses only the front camera for rendering and supervision
- freezes canonical Gaussian parameters by default
- trains dynamic modules in three stages:
  - `B`: `HexPlane/Grid`
  - `C`: `ODE + VideoEncoder`
  - `D`: joint dynamic fine-tuning
- periodically updates `deformation_table` with the original thresholding rule

## Run

```bash
python motion_main.py --config configs/stag4d_motion.yaml \
  canonical_dir=/path/to/canonical_dir \
  video_path=/path/to/front_video_frames \
  save_path=my_motion_run
```

If you only have a ply file:

```bash
python motion_main.py --config configs/stag4d_motion.yaml \
  canonical_ply=/path/to/model.ply \
  video_path=/path/to/front_video_frames \
  save_path=my_motion_run
```

## Notes

- `train_canonical` is `False` by default.
- validation renders are written under `valid/<save_path>/`.
- checkpoints are written under `logs/<save_path><step>/`.
