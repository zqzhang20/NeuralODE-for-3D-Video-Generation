import os
import fnmatch
import shutil
import argparse
from omegaconf import OmegaConf


def sort_key(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (0, int(digits), stem)
    return (1, stem)


def rename_images(input_dir, image_ext="png", zero_pad=3, copy=False):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_files = []
    for name in os.listdir(input_dir):
        lower_name = name.lower()
        if any(fnmatch.fnmatch(lower_name, pattern) for pattern in patterns):
            image_files.append(name)

    if len(image_files) == 0:
        raise RuntimeError(f"No image files found in: {input_dir}")

    image_files = sorted(image_files, key=sort_key)

    temp_paths = []
    for idx, old_name in enumerate(image_files):
        old_path = os.path.join(input_dir, old_name)
        temp_name = f"__tmp_rename_{idx:06d}{os.path.splitext(old_name)[1].lower()}"
        temp_path = os.path.join(input_dir, temp_name)
        if old_path != temp_path:
            os.replace(old_path, temp_path)
        temp_paths.append(temp_path)

    for idx, temp_path in enumerate(temp_paths):
        new_name = f"{idx:0{zero_pad}d}.{image_ext}"
        new_path = os.path.join(input_dir, new_name)
        if copy:
            shutil.copy2(temp_path, new_path)
        else:
            os.replace(temp_path, new_path)

    if copy:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    print(
        f"Renamed {len(temp_paths)} images in {input_dir} "
        f"to sequential names using .{image_ext} with zero_pad={zero_pad}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the yaml config file.")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    rename_images(
        input_dir=str(opt.input_dir),
        image_ext=str(getattr(opt, "image_ext", "png")),
        zero_pad=int(getattr(opt, "zero_pad", 3)),
        copy=bool(getattr(opt, "copy", False)),
    )
