import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate binary masks by treating near-black pixels as background."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source frames.")
    parser.add_argument("--output_dir", required=True, help="Directory to save generated masks.")
    parser.add_argument(
        "--glob",
        default="*.png",
        help="Glob pattern for input frames. Default: *.png",
    )
    parser.add_argument(
        "--black_threshold",
        type=int,
        default=20,
        help="Pixels with all channels <= this value are treated as background. Default: 20",
    )
    return parser.parse_args()


def build_mask(image_bgr: np.ndarray, black_threshold: int) -> np.ndarray:
    foreground = np.any(image_bgr > black_threshold, axis=-1)
    return (foreground.astype(np.uint8) * 255)


def iter_frames(input_dir: Path, pattern: str):
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file():
            yield path


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list(iter_frames(input_dir, args.glob))
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched {args.glob} in {input_dir}")

    for idx, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {frame_path}")

        mask = build_mask(
            image,
            black_threshold=args.black_threshold,
        )

        out_name = f"{frame_path.stem}_mask.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), mask)

        if idx < 10:
            vis_ratio = float(mask.mean() / 255.0)
            print(f"[mask] frame={frame_path.name} out={out_name} vis_ratio={vis_ratio:.6f}")

    print(f"Generated {len(frame_paths)} masks in {output_dir}")


if __name__ == "__main__":
    main()
