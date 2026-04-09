import os
import cv2
import argparse
from omegaconf import OmegaConf


def extract_frames(video_path, output_dir, image_ext="png", start=0, end=None, stride=1, zero_pad=3):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end is None:
        end = total_frames - 1
    end = min(end, total_frames - 1)
    start = max(0, start)
    stride = max(1, stride)

    frame_idx = 0
    saved_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < start:
            frame_idx += 1
            continue
        if frame_idx > end:
            break
        if (frame_idx - start) % stride != 0:
            frame_idx += 1
            continue

        out_name = f"{saved_idx:0{zero_pad}d}.{image_ext}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, frame)

        saved_idx += 1
        frame_idx += 1

    cap.release()
    print(
        f"Extracted {saved_idx} frames from {video_path} "
        f"into {output_dir} (source frames {start} to {end}, stride={stride})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the yaml config file.")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    extract_frames(
        video_path=str(opt.video),
        output_dir=str(opt.output),
        image_ext=str(getattr(opt, "image_ext", "png")),
        start=int(getattr(opt, "start", 0)),
        end=None if getattr(opt, "end", None) is None else int(opt.end),
        stride=int(getattr(opt, "stride", 1)),
        zero_pad=int(getattr(opt, "zero_pad", 3)),
    )
