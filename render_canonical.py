import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from cam_utils import OrbitCamera, orbit_camera
from gs_renderer_4d import MiniCam, Renderer, getProjectionMatrix


def to_uint8_image(tensor):
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (array * 255.0).round().astype(np.uint8)


def to_uint8_mask(tensor):
    if tensor.ndim == 3:
        tensor = tensor[0]
    array = tensor.detach().clamp(0, 1).cpu().numpy()
    return (array * 255.0).round().astype(np.uint8)


class MiniCamNoRectify:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, time=0.0):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.time = time
        self.c2w = torch.tensor(c2w, dtype=torch.float32, device="cuda")
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32, device="cuda").transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")
        self.right = self.c2w[:3, 0]
        self.up = self.c2w[:3, 1]


def save_render_outputs(outdir, prefix, out):
    rgb = to_uint8_image(out["image"])
    alpha = to_uint8_mask(out["alpha"])
    alpha_rgb = np.repeat(alpha[..., None], 3, axis=2)
    overlay = ((0.65 * rgb.astype(np.float32)) + (0.35 * alpha_rgb.astype(np.float32))).clip(0, 255).astype(np.uint8)

    imageio.imwrite(os.path.join(outdir, f"{prefix}_rgb.png"), rgb)
    imageio.imwrite(os.path.join(outdir, f"{prefix}_alpha.png"), alpha)
    imageio.imwrite(os.path.join(outdir, f"{prefix}_overlay.png"), overlay)

    radii = out["radii"]
    visible_count = int((radii > 0).sum().item())
    total_count = int(radii.numel())
    alpha_tensor = out["alpha"]
    print(f"[{prefix}] alpha_mean={float(alpha_tensor.mean().item()):.6f}")
    print(f"[{prefix}] alpha_max={float(alpha_tensor.max().item()):.6f}")
    print(f"[{prefix}] visible_gaussians={visible_count}/{total_count}")


def resolve_canonical_ply(opt):
    canonical_ply = str(getattr(opt, "canonical_ply", "") or "").strip()
    if canonical_ply:
        return canonical_ply
    canonical_dir = str(getattr(opt, "canonical_dir", "") or "").strip()
    if canonical_dir:
        candidate = os.path.join(canonical_dir, "model.ply")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Set `canonical_ply` or `canonical_dir` to an existing canonical 3DGS model.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--outdir", default="debug/canonical_render", help="Output directory")
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth in degrees")
    parser.add_argument("--elevation", type=float, default=None, help="Camera elevation in degrees")
    parser.add_argument("--radius", type=float, default=None, help="Camera radius")
    parser.add_argument("--fovy", type=float, default=None, help="Vertical FoV in degrees")
    parser.add_argument("--ref_size", type=int, default=None, help="Render resolution")
    parser.add_argument("--time", type=float, default=0.0, help="Render time for debugging")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    elevation = float(args.elevation if args.elevation is not None else getattr(opt, "elevation", 0.0))
    radius = float(args.radius if args.radius is not None else getattr(opt, "radius", 2.0))
    fovy = float(args.fovy if args.fovy is not None else getattr(opt, "fovy", 49.1))
    ref_size = int(args.ref_size if args.ref_size is not None else getattr(opt, "ref_size", 512))
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    renderer = Renderer(sh_degree=int(getattr(opt, "sh_degree", 0)))
    renderer.set_fov_gate_options(
        enabled=bool(getattr(opt, "enable_fov_gating", False)),
        margin=float(getattr(opt, "fov_gate_margin", 0.0)),
        scale_mult=float(getattr(opt, "fov_gate_scale_mult", 3.0)),
    )
    canonical_ply = resolve_canonical_ply(opt)
    renderer.initialize(input=canonical_ply)

    cam = OrbitCamera(ref_size, ref_size, r=radius, fovy=fovy)
    pose = orbit_camera(elevation, args.azimuth, radius)
    mini_cam = MiniCam(
        pose,
        ref_size,
        ref_size,
        cam.fovy,
        cam.fovx,
        cam.near,
        cam.far,
        time=args.time,
    )
    mini_cam_no_rectify = MiniCamNoRectify(
        pose,
        ref_size,
        ref_size,
        cam.fovy,
        cam.fovx,
        cam.near,
        cam.far,
        time=args.time,
    )

    with torch.no_grad():
        out_rectified = renderer.render(mini_cam, stage="coarse")
        out_no_rectify = renderer.render(mini_cam_no_rectify, stage="coarse")

    print(f"canonical_ply={canonical_ply}")
    print(f"output_dir={outdir}")
    print(f"camera=elev:{elevation:.3f} azim:{args.azimuth:.3f} radius:{radius:.3f} fovy:{fovy:.3f} size:{ref_size}")
    print("saved_views=rectified,no_rectify")
    save_render_outputs(outdir, "rectified", out_rectified)
    save_render_outputs(outdir, "no_rectify", out_no_rectify)


if __name__ == "__main__":
    main()
