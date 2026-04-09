import argparse
import os

import numpy as np
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement


SYSTEMS = {
    # From the shared SHARP camera model:
    # +x right, +y down, +z forward/into the screen.
    "sharp": "SHARP world/camera convention",
    # Current project's effective renderer convention after MiniCam rectification.
    "stag4d_rectified": "Current STAG4D renderer convention",
}

PRESETS = {
    "identity": np.eye(3, dtype=np.float32),
    # Rotate 180 degrees around the world x axis.
    "sharp_to_current": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    ),
    "flip_x": np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    "flip_y": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    "flip_z": np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    ),
}

SYSTEM_MAPPINGS = {
    ("sharp", "stag4d_rectified"): PRESETS["sharp_to_current"],
    ("stag4d_rectified", "sharp"): PRESETS["sharp_to_current"],
}


def parse_matrix(values):
    if values is None:
        return None
    if len(values) != 9:
        raise ValueError("--matrix expects 9 float values")
    return np.asarray([float(v) for v in values], dtype=np.float32).reshape(3, 3)


def axis_rotation_matrix(axis, angle_deg):
    angle = np.deg2rad(float(angle_deg))
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == "x":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.float32,
        )
    if axis == "y":
        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=np.float32,
        )
    if axis == "z":
        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported axis: {axis}")


def transform_vectors(data, names, matrix, scale):
    if not all(name in data.dtype.names for name in names):
        return
    vectors = np.stack([np.asarray(data[name], dtype=np.float32) for name in names], axis=1)
    transformed = vectors @ matrix.T
    if scale is not None:
        transformed = transformed * float(scale)
    for idx, name in enumerate(names):
        data[name] = transformed[:, idx].astype(data.dtype[name])


def as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def resolve_mapping(from_system, to_system):
    if from_system == to_system:
        return np.eye(3, dtype=np.float32), f"{from_system}_to_{to_system}"
    key = (from_system, to_system)
    if key not in SYSTEM_MAPPINGS:
        raise ValueError(f"No built-in mapping from {from_system} to {to_system}")
    return SYSTEM_MAPPINGS[key].copy(), f"{from_system}_to_{to_system}"


def apply_post_alignment(xyz, mode):
    mode = str(mode or "none").strip().lower()
    if mode in {"", "none"}:
        return xyz, np.zeros((1, 3), dtype=np.float32), "none"
    if mode == "center_z_to_positive_double_distance":
        center = xyz.mean(axis=0, keepdims=True)
        d = float(abs(center[0, 2]))
        offset = np.array([[0.0, 0.0, 1.6*d]], dtype=np.float32)
        return xyz + offset, offset, f"+z_by_2|center_z| (d={d:.6f})"
    raise ValueError(f"Unsupported post alignment mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Transform Gaussian PLY coordinates to match the current renderer.")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    input_path = str(getattr(opt, "input", "") or "").strip()
    output_path = str(getattr(opt, "output", "") or "").strip()
    if not input_path or not output_path:
        raise ValueError("Both `input` and `output` must be set in the yaml or CLI overrides.")

    from_system = str(getattr(opt, "from_system", "sharp"))
    to_system = str(getattr(opt, "to_system", "stag4d_rectified"))
    if from_system not in SYSTEMS:
        raise ValueError(f"Unknown from_system: {from_system}")
    if to_system not in SYSTEMS:
        raise ValueError(f"Unknown to_system: {to_system}")

    preset = str(getattr(opt, "preset", "sharp_to_current"))
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    matrix_values = getattr(opt, "matrix", None)
    if matrix_values is not None:
        matrix_values = list(matrix_values)
    rotate_axis = getattr(opt, "rotate_axis", None)
    if rotate_axis is not None:
        rotate_axis = str(rotate_axis)
    angle = float(getattr(opt, "angle", 180.0))
    scale = float(getattr(opt, "scale", 1.0))
    translate_values = getattr(opt, "translate", (0.0, 0.0, 0.0))
    translate = np.asarray(list(translate_values), dtype=np.float32).reshape(1, 3)
    center = as_bool(getattr(opt, "center", False))
    about_center = as_bool(getattr(opt, "about_center", False))
    transform_normals = as_bool(getattr(opt, "transform_normals", False))
    post_align = str(getattr(opt, "post_align", "center_z_to_positive_double_distance"))

    matrix = parse_matrix(matrix_values)
    transform_desc = "custom"
    if rotate_axis is not None:
        matrix = axis_rotation_matrix(rotate_axis, angle)
        transform_desc = f"rotate_{rotate_axis}_{angle:g}deg"
    elif matrix is None:
        use_system_mapping = as_bool(getattr(opt, "use_system_mapping", True), True)
        if use_system_mapping:
            matrix, transform_desc = resolve_mapping(from_system, to_system)
        else:
            matrix = PRESETS[preset]
            transform_desc = preset

    ply = PlyData.read(input_path)
    vertex = ply["vertex"]
    vertex_data = vertex.data.copy()

    xyz = np.stack(
        [
            np.asarray(vertex_data["x"], dtype=np.float32),
            np.asarray(vertex_data["y"], dtype=np.float32),
            np.asarray(vertex_data["z"], dtype=np.float32),
        ],
        axis=1,
    )

    xyz_mean_before = xyz.mean(axis=0)
    xyz_min_before = xyz.min(axis=0)
    xyz_max_before = xyz.max(axis=0)

    pivot = np.zeros((1, 3), dtype=np.float32)
    if about_center:
        pivot = xyz_mean_before.reshape(1, 3)
        xyz = xyz - pivot
    if center:
        xyz = xyz - xyz_mean_before.reshape(1, 3)
    xyz = xyz @ matrix.T
    if about_center:
        xyz = xyz + pivot
    xyz = xyz * scale
    xyz, post_offset, post_desc = apply_post_alignment(xyz, post_align)
    xyz = xyz + translate

    for idx, name in enumerate(("x", "y", "z")):
        vertex_data[name] = xyz[:, idx].astype(vertex_data.dtype[name])

    if transform_normals:
        transform_vectors(vertex_data, ("nx", "ny", "nz"), matrix, None)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out_vertex = PlyElement.describe(vertex_data, "vertex")
    out_ply = PlyData([out_vertex], text=ply.text, byte_order=ply.byte_order)
    out_ply.write(output_path)

    xyz_mean_after = xyz.mean(axis=0)
    xyz_min_after = xyz.min(axis=0)
    xyz_max_after = xyz.max(axis=0)

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"from_system={from_system}")
    print(f"to_system={to_system}")
    print(f"transform={transform_desc}")
    print(f"matrix=\n{matrix}")
    print(f"post_align={post_desc}")
    print(f"post_align_offset={post_offset.reshape(-1).tolist()}")
    print(f"scale={scale}")
    print(f"translate={translate.reshape(-1).tolist()}")
    print(f"center={center}")
    print(f"about_center={about_center}")
    print(f"xyz_mean_before={xyz_mean_before.tolist()}")
    print(f"xyz_mean_after={xyz_mean_after.tolist()}")
    print(f"xyz_min_before={xyz_min_before.tolist()}")
    print(f"xyz_max_before={xyz_max_before.tolist()}")
    print(f"xyz_min_after={xyz_min_after.tolist()}")
    print(f"xyz_max_after={xyz_max_after.tolist()}")


if __name__ == "__main__":
    main()
