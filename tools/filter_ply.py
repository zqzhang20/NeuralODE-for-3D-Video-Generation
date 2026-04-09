"""Filter a SHARP-generated PLY by a red foreground mask from its source view.

This script:
1. Reads SHARP metadata (intrinsics / image size / extrinsics) from the PLY.
2. Projects Gaussian centers back to the original source image view.
3. Keeps only the nearest visible Gaussian per pixel (simple z-buffer).
4. Filters visible Gaussians by a red foreground mask from the source image.
5. Randomly downsamples the remaining Gaussians to a target count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement


def load_red_foreground_mask(
    image_path: str | Path,
    image_width: int | None = None,
    image_height: int | None = None,
    red_threshold: int = 80,
    dominance: int = 40,
    black_threshold: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_width is not None and image_height is not None:
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)

    not_black = (r > black_threshold) | (g > black_threshold) | (b > black_threshold)
    red_dominant = (r > red_threshold) & ((r - g) > dominance) & ((r - b) > dominance)
    mask = (not_black & red_dominant).astype(np.uint8)
    return mask, image


def read_sharp_camera_from_ply(ply: PlyData) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    supplement_data: dict[str, np.ndarray] = {}
    supplement_keys = {"intrinsic", "image_size", "extrinsic"}

    for element in ply.elements:
        if element.name == "vertex":
            continue
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    if "intrinsic" not in supplement_data:
        raise ValueError("PLY does not contain SHARP intrinsics metadata.")

    intrinsic_data = supplement_data["intrinsic"]
    if intrinsic_data.size == 9:
        intrinsic = intrinsic_data.reshape(3, 3).astype(np.float32)
    elif intrinsic_data.size == 4:
        fx, fy, width, height = intrinsic_data.astype(np.float32)
        intrinsic = np.array(
            [
                [fx, 0.0, width * 0.5],
                [0.0, fy, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported intrinsic metadata size: {intrinsic_data.size}")

    if "image_size" in supplement_data:
        image_size = supplement_data["image_size"]
        width = int(image_size[0])
        height = int(image_size[1])
    elif intrinsic_data.size == 4:
        width = int(intrinsic_data[2])
        height = int(intrinsic_data[3])
    else:
        raise ValueError("PLY does not contain image_size metadata.")

    extrinsic_data = supplement_data.get("extrinsic", np.eye(4, dtype=np.float32).reshape(-1))
    if extrinsic_data.size == 16:
        extrinsic = extrinsic_data.reshape(4, 4).astype(np.float32)
    elif extrinsic_data.size == 12:
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3] = extrinsic_data.reshape(3, 4)
        extrinsic[:3, :3] = extrinsic[:3, :3].T
    else:
        raise ValueError(f"Unsupported extrinsic metadata size: {extrinsic_data.size}")

    return intrinsic, extrinsic, (width, height)


def project_points_to_source_view(
    xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3]

    xyz_cam = xyz @ rot.T + trans
    z = xyz_cam[:, 2]
    valid = z > 1e-8

    uvw = xyz_cam @ intrinsic.T
    u = np.zeros(xyz.shape[0], dtype=np.float32)
    v = np.zeros(xyz.shape[0], dtype=np.float32)
    u[valid] = uvw[valid, 0] / uvw[valid, 2]
    v[valid] = uvw[valid, 1] / uvw[valid, 2]
    return u, v, z.astype(np.float32), valid


def compute_visible_points(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    px = np.round(u).astype(np.int32)
    py = np.round(v).astype(np.int32)

    inside = valid & (px >= 0) & (px < image_width) & (py >= 0) & (py < image_height)
    inside_indices = np.where(inside)[0]
    if inside_indices.size == 0:
        return np.zeros_like(valid, dtype=bool)

    inside_px = px[inside_indices]
    inside_py = py[inside_indices]
    inside_z = z[inside_indices]

    pixel_ids = inside_py.astype(np.int64) * image_width + inside_px.astype(np.int64)
    order = np.argsort(inside_z, kind="stable")
    sorted_indices = inside_indices[order]
    sorted_pixel_ids = pixel_ids[order]

    _, first_positions = np.unique(sorted_pixel_ids, return_index=True)
    visible_indices = sorted_indices[first_positions]

    visible = np.zeros_like(valid, dtype=bool)
    visible[visible_indices] = True
    return visible


def filter_sharp_ply_by_mask(
    input_ply: str | Path,
    input_image: str | Path,
    output_ply: str | Path,
    image_width: int | None = None,
    image_height: int | None = None,
    red_threshold: int = 80,
    dominance: int = 40,
    black_threshold: int = 30,
    target_points: int = 5000,
    random_seed: int = 0,
) -> None:
    input_ply = Path(input_ply)
    output_ply = Path(output_ply)
    if not input_ply.is_file():
        raise FileNotFoundError(f"PLY not found: {input_ply}")

    ply = PlyData.read(str(input_ply))
    vertex = next((element.data for element in ply.elements if element.name == "vertex"), None)
    if vertex is None:
        raise ValueError("PLY file does not contain a vertex element.")

    intrinsic, extrinsic, ply_image_size = read_sharp_camera_from_ply(ply)

    override_width = image_width
    override_height = image_height
    mask, image_rgb = load_red_foreground_mask(
        input_image,
        image_width=override_width,
        image_height=override_height,
        red_threshold=red_threshold,
        dominance=dominance,
        black_threshold=black_threshold,
    )
    image_h, image_w = image_rgb.shape[:2]

    if override_width is not None or override_height is not None:
        src_w, src_h = ply_image_size
        sx = image_w / float(src_w)
        sy = image_h / float(src_h)
        intrinsic = intrinsic.copy()
        intrinsic[0, 0] *= sx
        intrinsic[1, 1] *= sy
        intrinsic[0, 2] *= sx
        intrinsic[1, 2] *= sy
    elif (image_w, image_h) != ply_image_size:
        raise ValueError(
            "Input image size does not match the SHARP PLY metadata. "
            f"image={image_w}x{image_h}, ply={ply_image_size[0]}x{ply_image_size[1]}. "
            "Either use the original image or set image_width/image_height explicitly."
        )

    xyz = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    count_before = xyz.shape[0]

    u, v, z, valid = project_points_to_source_view(xyz, intrinsic, extrinsic)
    visible = compute_visible_points(u, v, z, valid, image_w, image_h)

    px = np.round(u).astype(np.int32)
    py = np.round(v).astype(np.int32)
    keep = np.zeros(count_before, dtype=bool)
    visible_indices = np.where(visible)[0]
    keep[visible_indices] = mask[py[visible_indices], px[visible_indices]] > 0

    count_visible = int(visible.sum())
    count_mask = int(keep.sum())
    if count_mask == 0:
        raise RuntimeError(
            "Mask filtering removed all visible points. Check the mask thresholds or the input image."
        )

    if count_mask > int(target_points):
        rng = np.random.default_rng(int(random_seed))
        selected = rng.choice(np.where(keep)[0], size=int(target_points), replace=False)
        final_keep = np.zeros_like(keep)
        final_keep[selected] = True
    else:
        final_keep = keep

    filtered_vertex = vertex[final_keep]
    count_after = len(filtered_vertex)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(filtered_vertex, "vertex")], text=ply.text).write(str(output_ply))

    print(f"Original points: {count_before}")
    print(f"Visible points in source view: {count_visible}")
    print(f"Visible points on red mask: {count_mask}")
    print(f"Points written to output: {count_after}")
    print(f"Saved filtered PLY to: {output_ply}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the yaml config file.")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    filter_sharp_ply_by_mask(
        input_ply=str(opt.input_ply),
        input_image=str(opt.input_image),
        output_ply=str(opt.output_ply),
        image_width=None if getattr(opt, "image_width", None) is None else int(opt.image_width),
        image_height=None if getattr(opt, "image_height", None) is None else int(opt.image_height),
        red_threshold=int(getattr(opt, "red_threshold", 80)),
        dominance=int(getattr(opt, "dominance", 40)),
        black_threshold=int(getattr(opt, "black_threshold", 30)),
        target_points=int(getattr(opt, "target_points", 5000)),
        random_seed=int(getattr(opt, "random_seed", 0)),
    )


if __name__ == "__main__":
    main()
