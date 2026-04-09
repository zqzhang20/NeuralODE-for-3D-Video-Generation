import os
import cv2
import time
import tqdm
import imageio
import json
import numpy as np
import fnmatch

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from cam_utils import orbit_camera, OrbitCamera
from dynamics_losses import anchor_shape_loss, curvature_loss, cycle_sparse_loss, velocity_reg
from gs_renderer_4d import MiniCam, Renderer


def save_image_to_local(image_tensor, file_path):
    image_tensor = image_tensor.clamp(0, 1)
    vutils.save_image(image_tensor, file_path)


def compute_mask_motion_stats(mask_tensor, eps=1e-6):
    if mask_tensor.ndim == 4:
        mask = mask_tensor[0, 0]
    elif mask_tensor.ndim == 3:
        mask = mask_tensor[0]
    else:
        mask = mask_tensor
    mask = mask.float().clamp(0.0, 1.0)
    h, w = mask.shape[-2:]
    ys = torch.linspace(0.0, 1.0, h, device=mask.device, dtype=mask.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=mask.device, dtype=mask.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    mass = mask.sum()
    norm = torch.clamp(mass, min=eps)
    center_x = (mask * grid_x).sum() / norm
    center_y = (mask * grid_y).sum() / norm
    area = mask.mean()
    center = torch.stack([center_x, center_y], dim=0)
    return center, area


class FrontVideoSequence:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.frame_paths = self._discover_frame_paths()
        self.input_img_torch_batch = []
        self.input_mask_torch_batch = []
        self.mask_vis_ratios = []
        self.frame_vis_scores = []
        self.frame_border_touches = []
        self.frame_border_side_touches = []
        self.mask_centers = []
        self.mask_areas = []
        self._load_frames()
        self.frames = len(self.input_img_torch_batch)
        self.time_denom = max(1, self.frames - 1)
        canonical_frame_idx = int(getattr(self.opt, "canonical_frame_idx", 12)) - 1
        self.canonical_frame_idx = int(np.clip(canonical_frame_idx, 0, max(0, self.frames - 1)))
        self._build_relative_buckets()
        self.cycle_pairs = self._build_cycle_pairs()
        self.frame_states = self._build_frame_states()
        self.motion_segments, self.frame_to_segment_id = self._build_motion_segments()

    def _discover_frame_paths(self):
        roots = []
        video_path = str(getattr(self.opt, "video_path", "") or "").strip()
        if video_path:
            roots.append(video_path)
        path = str(getattr(self.opt, "path", "") or "").strip()
        if path:
            roots.append(os.path.join(path, "ref"))
            roots.append(path)

        patterns = ("*_rgba.png", "*.png", "*.jpg", "*.jpeg")
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            matches = []
            for pattern in patterns:
                matches.extend(
                    os.path.join(root, name)
                    for name in os.listdir(root)
                    if fnmatch.fnmatch(name.lower(), pattern.lower())
                )
                if matches:
                    break
            filtered = [p for p in matches if os.path.isfile(p)]
            if filtered:
                return sorted(filtered, key=self._frame_sort_key)
        raise FileNotFoundError(
            "No video frames found. Set `video_path` to a directory containing front-view frames."
        )

    def _frame_sort_key(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return (0, int(digits), stem)
        return (1, stem)

    def _candidate_mask_paths(self, frame_path):
        frame_dir = os.path.dirname(frame_path)
        stem = os.path.splitext(os.path.basename(frame_path))[0]
        names = [stem + "_mask.png", stem.replace("_rgba", "") + "_mask.png", stem + ".png"]
        mask_dirs = []
        mask_path = str(getattr(self.opt, "mask_path", "") or "").strip()
        if mask_path:
            mask_dirs.append(mask_path)
        mask_dirs.append(os.path.join(frame_dir, "masks"))
        for mask_dir in mask_dirs:
            for name in names:
                yield os.path.join(mask_dir, name)

    def _load_frame(self, frame_path):
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame: {frame_path}")

        mask = None
        if img.ndim == 3 and img.shape[-1] == 4:
            mask = img[..., 3:].astype(np.float32) / 255.0
            img = img[..., :3]
        else:
            for candidate in self._candidate_mask_paths(frame_path):
                if os.path.exists(candidate):
                    mask_img = cv2.imread(candidate, cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        if mask_img.ndim == 3:
                            mask_img = mask_img[..., 0]
                        mask = mask_img.astype(np.float32) / 255.0
                        mask = mask[..., None]
                        break
        if mask is None:
            mask = np.ones((*img.shape[:2], 1), dtype=np.float32)

        img = cv2.resize(img, (self.opt.ref_size, self.opt.ref_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.opt.ref_size, self.opt.ref_size), interpolation=cv2.INTER_AREA)
        if mask.ndim == 2:
            mask = mask[..., None]
        img = img.astype(np.float32) / 255.0
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        image = img[..., :3] * mask + (1.0 - mask)
        image = image[..., ::-1].copy()
        image_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return image_t, mask_t

    def _compute_frame_visibility_score(self, mask_tensor):
        m = mask_tensor[0, 0]
        area_ratio = float(m.mean().item())
        active = m > 0.5
        active_count = int(active.sum().item())
        if active_count == 0:
            empty_sides = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
            return area_ratio, 0.0, 1.0, 0.0, empty_sides

        ys, xs = torch.where(active)
        y0, y1 = int(ys.min().item()), int(ys.max().item())
        x0, x1 = int(xs.min().item()), int(xs.max().item())
        bbox_h = max(1, y1 - y0 + 1)
        bbox_w = max(1, x1 - x0 + 1)
        bbox_area = float(bbox_h * bbox_w)
        bbox_fill = float(active_count / bbox_area)

        border = torch.zeros_like(active, dtype=torch.bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        border_touch = float((active & border).sum().item() / max(1, active_count))
        side_touches = {
            "left": float((active[:, 0]).sum().item() / max(1, active_count)),
            "right": float((active[:, -1]).sum().item() / max(1, active_count)),
            "top": float((active[0, :]).sum().item() / max(1, active_count)),
            "bottom": float((active[-1, :]).sum().item() / max(1, active_count)),
        }

        score = 0.35 * area_ratio + 0.65 * bbox_fill
        if border_touch > 0.05:
            score *= 0.85
        return area_ratio, bbox_fill, border_touch, float(np.clip(score, 0.0, 1.0)), side_touches

    def _load_frames(self):
        for frame_path in self.frame_paths:
            image_t, mask_t = self._load_frame(frame_path)
            self.input_img_torch_batch.append(image_t)
            self.input_mask_torch_batch.append(mask_t)
            self.mask_vis_ratios.append(float(mask_t.mean().item()))
            center, area = compute_mask_motion_stats(mask_t)
            self.mask_centers.append(center.detach())
            self.mask_areas.append(area.detach())
            _, _, border_touch, score, side_touches = self._compute_frame_visibility_score(mask_t)
            self.frame_vis_scores.append(score)
            self.frame_border_touches.append(border_touch)
            self.frame_border_side_touches.append(side_touches)

    def _build_relative_buckets(self):
        n = len(self.frame_vis_scores)
        order = np.argsort(-np.asarray(self.frame_vis_scores, dtype=np.float32))
        high_ratio = float(np.clip(getattr(self.opt, "high_frame_ratio", 0.3), 0.0, 1.0))
        mid_ratio = float(np.clip(getattr(self.opt, "mid_frame_ratio", 0.4), 0.0, 1.0))
        high_n = int(max(1, round(n * high_ratio)))
        mid_n = int(max(0, round(n * mid_ratio)))
        if high_n + mid_n > n:
            mid_n = max(0, n - high_n)
        self.high_frames = [int(i) for i in order[:high_n].tolist()]
        self.mid_frames = [int(i) for i in order[high_n:high_n + mid_n].tolist()]
        self.low_frames = [int(i) for i in order[high_n + mid_n:].tolist()]
        self.high_frame_set = set(self.high_frames)
        self.mid_frame_set = set(self.mid_frames)
        self.frame_rank_scores = [0.0 for _ in range(n)]
        if n == 1:
            self.frame_rank_scores[order[0]] = 1.0
        elif n > 1:
            denom = float(n - 1)
            for rank, frame_idx in enumerate(order.tolist()):
                self.frame_rank_scores[int(frame_idx)] = 1.0 - (float(rank) / denom)

    def _build_cycle_pairs(self):
        pairs = []
        low = np.array(self.low_frames, dtype=np.int64)
        high = np.array(self.high_frames, dtype=np.int64)
        if low.size == 0 or high.size == 0:
            return pairs
        for idx in low:
            before = high[high < idx]
            after = high[high > idx]
            if before.size > 0 and after.size > 0:
                pairs.append((int(before[-1]), int(after[0])))
        return pairs

    def _build_frame_states(self):
        if self.frames == 0:
            return []
        canonical_area = float(self.mask_areas[self.canonical_frame_idx].item())
        top_k = max(1, min(5, self.frames))
        top_score_idx = np.argsort(-np.asarray(self.frame_vis_scores, dtype=np.float32))[:top_k]
        ref_area = max(
            canonical_area,
            float(np.mean([float(self.mask_areas[int(idx)].item()) for idx in top_score_idx])),
            1e-6,
        )
        ref_score = max(
            float(self.frame_vis_scores[self.canonical_frame_idx]),
            float(np.mean([float(self.frame_vis_scores[int(idx)]) for idx in top_score_idx])),
            1e-6,
        )
        visible_ratio = float(getattr(self.opt, "relative_visible_ratio", 0.6))
        hidden_ratio = float(getattr(self.opt, "relative_hidden_ratio", 0.15))
        boundary_touch = float(getattr(self.opt, "border_touch_boundary", 0.10))
        hidden_touch = float(getattr(self.opt, "border_touch_hidden", 0.20))
        visible_score = float(getattr(self.opt, "relative_visible_score", 0.7))
        hidden_score = float(getattr(self.opt, "relative_hidden_score", 0.2))
        states = []
        for idx in range(self.frames):
            rel_area = float(self.mask_areas[idx].item()) / ref_area
            rel_score = float(self.frame_vis_scores[idx]) / ref_score
            border = float(self.frame_border_touches[idx])
            if rel_area < hidden_ratio or (rel_score < hidden_score and border >= hidden_touch):
                states.append("hidden")
            elif rel_area >= visible_ratio and rel_score >= visible_score and border < boundary_touch:
                states.append("visible")
            else:
                states.append("boundary")
        return states

    def _build_motion_segments(self):
        segments = []
        frame_to_segment_id = [-1 for _ in range(self.frames)]
        if self.frames == 0:
            return segments, frame_to_segment_id
        start = 0
        segment_id = 0
        while start < self.frames:
            state = self.frame_states[start]
            end = start
            while end + 1 < self.frames and self.frame_states[end + 1] == state:
                end += 1
            indices = list(range(start, end + 1))
            local_count = len(indices)
            if local_count == 1:
                local_times = [0.0]
            else:
                local_times = [float(i / (local_count - 1)) for i in range(local_count)]
            left_anchor_idx = start - 1 if start > 0 else None
            right_anchor_idx = end + 1 if end + 1 < self.frames else None
            if state == "hidden":
                if right_anchor_idx is not None:
                    ref_idx = int(right_anchor_idx)
                elif left_anchor_idx is not None:
                    ref_idx = int(left_anchor_idx)
                else:
                    ref_idx = int(indices[0])
            elif self.canonical_frame_idx in indices:
                ref_idx = self.canonical_frame_idx
            else:
                ref_idx = max(indices, key=lambda idx: float(self.frame_vis_scores[idx]))
            segment = {
                "segment_id": segment_id,
                "segment_type": state,
                "start_idx": start,
                "end_idx": end,
                "indices": indices,
                "ref_idx": int(ref_idx),
                "left_anchor_idx": left_anchor_idx,
                "right_anchor_idx": right_anchor_idx,
                "local_times": local_times,
            }
            segments.append(segment)
            for idx in indices:
                frame_to_segment_id[idx] = segment_id
            segment_id += 1
            start = end + 1
        return segments, frame_to_segment_id


class MotionTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(opt.device)
        self.sequence = FrontVideoSequence(opt, self.device)
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.cam = OrbitCamera(self.opt.ref_size, self.opt.ref_size, r=self.opt.radius, fovy=self.opt.fovy)

        self.input_img_torch_batch = self.sequence.input_img_torch_batch
        self.input_mask_torch_batch = self.sequence.input_mask_torch_batch
        self.mask_vis_ratios = self.sequence.mask_vis_ratios
        self.frame_vis_scores = self.sequence.frame_vis_scores
        self.frame_border_touches = self.sequence.frame_border_touches
        self.frame_border_side_touches = self.sequence.frame_border_side_touches
        self.mask_centers = self.sequence.mask_centers
        self.mask_areas = self.sequence.mask_areas
        self.frame_rank_scores = self.sequence.frame_rank_scores
        self.high_frames = self.sequence.high_frames
        self.mid_frames = self.sequence.mid_frames
        self.low_frames = self.sequence.low_frames
        self.high_frame_set = self.sequence.high_frame_set
        self.mid_frame_set = self.sequence.mid_frame_set
        self.low_frame_set = set(self.low_frames)
        self.frame_states = self.sequence.frame_states
        self.motion_segments = self.sequence.motion_segments
        self.frame_to_segment_id = self.sequence.frame_to_segment_id
        self.visible_frame_set = {idx for idx, state in enumerate(self.frame_states) if state == "visible"}
        self.boundary_frame_set = {idx for idx, state in enumerate(self.frame_states) if state == "boundary"}
        self.hidden_frame_set = {idx for idx, state in enumerate(self.frame_states) if state == "hidden"}
        self.motion_segments_by_id = {int(seg["segment_id"]): seg for seg in self.motion_segments}
        self.cycle_pairs = self.sequence.cycle_pairs
        self.frames = self.sequence.frames
        self.time_denom = self.sequence.time_denom
        self.canonical_frame_idx = self.sequence.canonical_frame_idx
        self.canonical_time = float(self.canonical_frame_idx / self.time_denom) if self.frames > 0 else 0.0
        self.video_frames_tensor = torch.cat(self.input_img_torch_batch, dim=0)
        ref_center = self.mask_centers[self.canonical_frame_idx]
        self.canonical_mask_template = self.input_mask_torch_batch[self.canonical_frame_idx].detach().clone().to(self.device)
        canonical_area = float(self.mask_areas[self.canonical_frame_idx].item())
        self.canonical_mask_radius_2d = max(np.sqrt(max(canonical_area, 1e-8) / np.pi), 1e-4)
        self.mask_center_motion = [
            float(torch.norm(center - ref_center, p=2).item()) for center in self.mask_centers
        ]
        max_center_motion = max(self.mask_center_motion) if len(self.mask_center_motion) > 0 else 0.0
        motion_eps = 1e-6
        self.mask_center_motion_weights = [
            1.0 + (motion / max(max_center_motion, motion_eps)) for motion in self.mask_center_motion
        ]
        motion_sampling = np.asarray(self.mask_center_motion_weights, dtype=np.float32)
        motion_sampling = np.square(motion_sampling)
        if self.frames > 0:
            for idx in range(self.frames):
                if idx in self.hidden_frame_set:
                    motion_sampling[idx] *= float(getattr(self.opt, "hidden_frame_sample_ratio", 0.2))
                elif idx in self.boundary_frame_set:
                    motion_sampling[idx] *= float(getattr(self.opt, "boundary_frame_sample_ratio", 1.5))
        motion_sampling_sum = float(motion_sampling.sum())
        if motion_sampling.size == 0 or motion_sampling_sum <= motion_eps:
            self.motion_sampling_probs = None
        else:
            motion_sampling = motion_sampling / motion_sampling_sum
            motion_sampling = motion_sampling.astype(np.float64)
            motion_sampling = motion_sampling / motion_sampling.sum()
            self.motion_sampling_probs = motion_sampling

        self.stage_traj_steps = int(getattr(self.opt, "stage_traj_steps", max(1, int(self.opt.iters * 0.25))))
        self.stage_ode_steps = int(getattr(self.opt, "stage_ode_steps", max(self.stage_traj_steps + 1, int(self.opt.iters * 0.5))))
        self.stage_hex_steps = int(getattr(self.opt, "stage_hex_steps", max(self.stage_ode_steps + 1, int(self.opt.iters * 0.8))))
        traj_boundary_start_ratio = float(getattr(self.opt, "traj_boundary_start_ratio", 0.4))
        traj_hidden_start_ratio = float(getattr(self.opt, "traj_hidden_start_ratio", 0.6))
        traj_global_smooth_start_ratio = float(getattr(self.opt, "traj_global_smooth_start_ratio", 0.85))
        self.traj_boundary_start_step = int(np.clip(round(self.stage_traj_steps * traj_boundary_start_ratio), 1, self.stage_traj_steps))
        self.traj_hidden_start_step = int(np.clip(round(self.stage_traj_steps * traj_hidden_start_ratio), 1, self.stage_traj_steps))
        self.traj_global_smooth_start_step = int(np.clip(round(self.stage_traj_steps * traj_global_smooth_start_ratio), 1, self.stage_traj_steps))
        self.hidden_traj_initialized = False
        self.observed_traj_initialized = False
        self.lambda_vel = float(getattr(self.opt, "lambda_vel", 0.001))
        self.lambda_curv = float(getattr(self.opt, "lambda_curv", 0.001))
        self.lambda_anchor = float(getattr(self.opt, "lambda_anchor", 0.001))
        self.lambda_cycle_sparse = float(getattr(self.opt, "lambda_cycle_sparse", 0.001))
        self.lambda_ode_center = float(getattr(self.opt, "lambda_ode_center", 0.0))
        self.lambda_ode_area = float(getattr(self.opt, "lambda_ode_area", 0.0))
        self.lambda_rgb = float(getattr(self.opt, "lambda_rgb", 20000.0))
        self.lambda_alpha = float(getattr(self.opt, "lambda_alpha", 5000.0))
        self.traj_rgb_factor = float(getattr(self.opt, "traj_rgb_factor", 0.25))
        self.traj_alpha_factor = float(getattr(self.opt, "traj_alpha_factor", 0.5))
        self.traj_center_factor = float(getattr(self.opt, "traj_center_factor", 5.0))
        self.traj_area_factor = float(getattr(self.opt, "traj_area_factor", 0.5))
        self.ode_rgb_factor = float(getattr(self.opt, "ode_rgb_factor", 0.1))
        self.ode_alpha_factor = float(getattr(self.opt, "ode_alpha_factor", 0.25))
        self.ode_vel_factor = float(getattr(self.opt, "ode_vel_factor", 0.0))
        self.ode_center_factor = float(getattr(self.opt, "ode_center_factor", 1.0))
        self.ode_area_factor = float(getattr(self.opt, "ode_area_factor", 1.0))
        self.boundary_rgb_factor = float(getattr(self.opt, "boundary_rgb_factor", 0.5))
        self.boundary_alpha_factor = float(getattr(self.opt, "boundary_alpha_factor", 1.0))
        self.boundary_area_factor = float(getattr(self.opt, "boundary_area_factor", 0.5))
        self.boundary_center_factor = float(getattr(self.opt, "boundary_center_factor", 1.5))
        self.hidden_rgb_factor = float(getattr(self.opt, "hidden_rgb_factor", 0.1))
        self.hidden_alpha_factor = float(getattr(self.opt, "hidden_alpha_factor", 0.5))
        self.hidden_area_factor = float(getattr(self.opt, "hidden_area_factor", 0.25))
        self.hidden_center_factor = float(getattr(self.opt, "hidden_center_factor", 1.5))
        self.boundary_smoothness_factor = float(getattr(self.opt, "boundary_smoothness_factor", 0.35))
        self.hidden_smoothness_factor = float(getattr(self.opt, "hidden_smoothness_factor", 0.15))
        self.border_smoothness_scale = float(getattr(self.opt, "border_smoothness_scale", 0.75))
        self.visibility_smoothness_floor = float(getattr(self.opt, "visibility_smoothness_floor", 0.2))
        self.boundary_event_boost = float(getattr(self.opt, "boundary_event_boost", 1.0))
        self.boundary_direction_factor = float(getattr(self.opt, "boundary_direction_factor", 0.5))
        self.boundary_observation_floor = float(getattr(self.opt, "boundary_observation_floor", 0.1))
        self.hidden_bridge_hermite_factor = float(getattr(self.opt, "hidden_bridge_hermite_factor", 0.35))
        self.hidden_bridge_endpoint_factor = float(getattr(self.opt, "hidden_bridge_endpoint_factor", 1.0))
        self.hidden_bridge_trend_factor = float(getattr(self.opt, "hidden_bridge_trend_factor", 0.5))
        self.hidden_exit_margin_factor = float(getattr(self.opt, "hidden_exit_margin_factor", 1.1))
        self.traj_boundary_reliable_threshold = float(getattr(self.opt, "traj_boundary_reliable_threshold", 0.2))
        self.traj_boundary_min_area = float(getattr(self.opt, "traj_boundary_min_area", 0.05))
        self.traj_boundary_max_border = float(getattr(self.opt, "traj_boundary_max_border", 0.35))
        self.traj_boundary_anchor_factor = float(getattr(self.opt, "traj_boundary_anchor_factor", 2.0))
        self.traj_segment_transition_factor = float(getattr(self.opt, "traj_segment_transition_factor", 0.25))
        self.traj_global_transition_factor = float(getattr(self.opt, "traj_global_transition_factor", 0.2))
        self.traj_global_smooth_factor = float(getattr(self.opt, "traj_global_smooth_factor", 0.2))
        self.traj_boundary_area_factor = float(getattr(self.opt, "traj_boundary_area_factor", 1.0))
        self.traj_boundary_one_sided_factor = float(getattr(self.opt, "traj_boundary_one_sided_factor", 2.0))
        self.hidden_damping_factor = float(getattr(self.opt, "hidden_damping_factor", 1.0))
        self.hidden_damping_floor = float(getattr(self.opt, "hidden_damping_floor", 0.25))
        self.lambda_traj_boundary = float(getattr(self.opt, "lambda_traj_boundary", 20.0))
        self.lambda_traj_center = float(getattr(self.opt, "lambda_traj_center", 50.0))
        self.lambda_traj_smooth = float(getattr(self.opt, "lambda_traj_smooth", 5.0))
        self.lambda_segment_transition = float(getattr(self.opt, "lambda_segment_transition", 10.0))
        self.lambda_canonical_traj = float(getattr(self.opt, "lambda_canonical_traj", 10.0))
        self.lambda_hidden_bridge = float(getattr(self.opt, "lambda_hidden_bridge", 50.0))
        self.lambda_hidden_velocity = float(getattr(self.opt, "lambda_hidden_velocity", 10.0))
        self.lambda_hidden_energy = float(getattr(self.opt, "lambda_hidden_energy", 2.0))
        self.lambda_hidden_damping = float(getattr(self.opt, "lambda_hidden_damping", 10.0))
        self.current_train_stage = "TRAJ"
        self.step = 0
        self.optimizer = None
        self.motion_debug_interval = int(getattr(self.opt, "motion_debug_interval", 100))
        self.traj_vis_interval = int(getattr(self.opt, "traj_vis_interval", 0))
        self.traj_render_fov_scale = float(getattr(self.opt, "traj_render_fov_scale", 2.8))
        self.traj_render_size = int(getattr(self.opt, "traj_render_size", int(self.opt.ref_size)))
        self.ode_diag_fov_scale = float(getattr(self.opt, "ode_diag_fov_scale", self.traj_render_fov_scale))
        self.ode_diag_size = int(getattr(self.opt, "ode_diag_size", self.traj_render_size))
        self.ode_delta_draw_points = int(getattr(self.opt, "ode_delta_draw_points", 5000))
        self.log_dir = os.path.join(self.opt.outdir, self.opt.save_path)
        os.makedirs(self.log_dir, exist_ok=True)
        traj_vis_dir = str(getattr(self.opt, "traj_vis_dir", "") or "").strip()
        if traj_vis_dir:
            self.traj_vis_root = traj_vis_dir if os.path.isabs(traj_vis_dir) else os.path.join(self.opt.outdir, traj_vis_dir)
        else:
            self.traj_vis_root = os.path.join(self.opt.outdir, self.opt.save_path, "traj_vis")
        os.makedirs(self.traj_vis_root, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "train.log")
        self._log_fh = open(self.log_path, "a", encoding="ascii")

    def _log(self, message):
        print(message)
        self._log_fh.write(message + "\n")
        self._log_fh.flush()

    def _resolve_motion_mode(self, stage_name):
        if stage_name == "TRAJ":
            return "traj"
        if stage_name == "ODE":
            return "ode"
        if stage_name == "HEX":
            return "hex"
        return "joint"

    def _resolve_canonical_ply(self):
        canonical_ply = str(getattr(self.opt, "canonical_ply", "") or "").strip()
        if canonical_ply:
            return canonical_ply
        canonical_dir = str(getattr(self.opt, "canonical_dir", "") or "").strip()
        if canonical_dir:
            candidate = os.path.join(canonical_dir, "model.ply")
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError("Set `canonical_ply` or `canonical_dir` to an existing canonical 3DGS model.")

    def prepare_train(self):
        canonical_ply = self._resolve_canonical_ply()
        self.renderer.initialize(input=canonical_ply)

        canonical_dir = str(getattr(self.opt, "canonical_dir", "") or "").strip()
        resume_dynamic = bool(getattr(self.opt, "resume_dynamic", False))
        if resume_dynamic and canonical_dir and os.path.exists(os.path.join(canonical_dir, "deformation.pth")):
            self.renderer.gaussians.load_model(canonical_dir)

        self.renderer.gaussians._deformation.set_motion_segments(self.motion_segments, self.frame_to_segment_id)
        self.renderer.gaussians.training_setup(self.opt)
        self.renderer.gaussians._deformation.set_ode_steps(getattr(self.opt, "ode_steps", 8))
        self.renderer.gaussians._deformation.set_ode_grad_frame_window(getattr(self.opt, "ode_grad_frame_window", 2))
        self.renderer.gaussians._deformation.set_vmax(getattr(self.opt, "vmax", 1.0))
        self.renderer.gaussians._deformation.set_opacity_vmax(getattr(self.opt, "opacity_vmax", 2.0))
        self.renderer.gaussians._deformation.set_anchor_depth_mode(getattr(self.opt, "anchor_depth_mode", "fixed"))
        self.renderer.gaussians._deformation.set_hex_affects_xyz(bool(getattr(self.opt, "hex_affects_xyz", False)))
        self.renderer.gaussians._deformation.set_canonical_time(self.canonical_time)
        self.renderer.gaussians.set_deformation_table_usage(bool(getattr(self.opt, "use_deformation_table", False)))
        physical_full_render = bool(getattr(self.opt, "physically_render_full_gaussians", True))
        self.renderer.set_fov_gate_options(
            enabled=bool(getattr(self.opt, "enable_fov_gating", False)) and (not physical_full_render),
            margin=float(getattr(self.opt, "fov_gate_margin", 0.0)),
            scale_mult=float(getattr(self.opt, "fov_gate_scale_mult", 3.0)),
        )
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        self.canonical_alignment_offset_2d = torch.zeros(2, device=self.device)
        self.canonical_alignment_offset_world = torch.zeros(3, device=self.device)
        if bool(getattr(self.opt, "align_canonical_center", True)):
            self._align_canonical_template_to_frame_center()

    @torch.no_grad()
    def _initialize_observed_segment_trajectories_from_gt(self):
        deformation = self.renderer.gaussians._deformation
        for segment in self.motion_segments:
            if segment["segment_type"] not in ("visible", "boundary"):
                continue
            target = torch.stack([self._global_center_delta(idx)[:2] for idx in segment["indices"]], dim=0).detach()
            if target.shape[0] == 0:
                continue
            deformation.set_segment_control_points(int(segment["segment_id"]), target)

    @torch.no_grad()
    def _initialize_hidden_segment_trajectories(self, use_gt_boundary=False):
        deformation = self.renderer.gaussians._deformation
        for segment in self.motion_segments:
            if segment["segment_type"] != "hidden":
                continue
            bridge = self._hidden_bridge_targets(segment, use_gt_boundary=use_gt_boundary).detach()
            if bridge.shape[0] == 0:
                continue
            deformation.set_segment_control_points(int(segment["segment_id"]), bridge)

    @torch.no_grad()
    def _sync_hidden_segment_trajectories_to_bridge(self, use_gt_boundary=False):
        self._initialize_hidden_segment_trajectories(use_gt_boundary=use_gt_boundary)


    def _traj_hidden_enabled(self):
        return self.current_train_stage == "TRAJ" and self.step >= self.traj_hidden_start_step

    def _traj_boundary_enabled(self):
        return self.current_train_stage == "TRAJ" and self.step < self.traj_hidden_start_step

    def _traj_global_smooth_enabled(self):
        return self.current_train_stage == "TRAJ" and self.step >= self.traj_global_smooth_start_step

    def _traj_visible_phase_active(self):
        return False

    def _traj_boundary_phase_active(self):
        return self.current_train_stage == "TRAJ" and self.step < self.traj_hidden_start_step

    def _traj_hidden_phase_active(self):
        return self.current_train_stage == "TRAJ" and self.traj_hidden_start_step <= self.step < self.traj_global_smooth_start_step

    def _traj_global_phase_active(self):
        return self.current_train_stage == "TRAJ" and self.step >= self.traj_global_smooth_start_step

    def _set_traj_segment_trainability(self):
        deformation = self.renderer.gaussians._deformation.deformation_net.segment_translation_trajectory
        boundary_phase = self._traj_boundary_phase_active()
        for seg in self.motion_segments:
            seg_id = str(int(seg["segment_id"]))
            traj = deformation.trajectories[seg_id]
            active = False
            if seg["segment_type"] == "boundary":
                active = boundary_phase
            traj.translations.requires_grad_(active)

    def _resolve_stage_name(self):
        if self.step < self.stage_traj_steps:
            return "TRAJ"
        if self.step < self.stage_ode_steps:
            return "ODE"
        if self.step < self.stage_hex_steps:
            return "HEX"
        return "JOINT"

    def _sample_frame_for_stage(self, stage_name):
        if stage_name == "TRAJ":
            if self._traj_hidden_enabled():
                pool = list(range(self.frames))
            else:
                pool = [idx for idx in range(self.frames) if idx not in self.hidden_frame_set]
                if len(pool) == 0:
                    pool = list(range(self.frames))
            if self.motion_sampling_probs is not None and len(pool) == self.frames:
                return int(np.random.choice(self.frames, p=self.motion_sampling_probs))
            return int(np.random.choice(pool))
        if stage_name == "HEX":
            pool = [idx for idx in (self.high_frames + self.mid_frames) if idx not in self.hidden_frame_set]
            if len(pool) == 0:
                pool = [idx for idx in range(self.frames) if idx not in self.hidden_frame_set]
            return int(np.random.choice(pool))
        if stage_name == "ODE" and self.frames > 0 and self.motion_sampling_probs is not None:
            return int(np.random.choice(self.frames, p=self.motion_sampling_probs))
        if stage_name == "JOINT" and self.frames > 0 and self.motion_sampling_probs is not None and np.random.rand() < 0.6:
            return int(np.random.choice(self.frames, p=self.motion_sampling_probs))
        else:
            pool = list(range(self.frames))
        if len(pool) == 0:
            pool = list(range(self.frames))
        return int(np.random.choice(pool))

    def _segment_translation(self, frame_idx):
        return self.renderer.gaussians._deformation.get_segment_translation(frame_idx).reshape(-1).to(self.device)

    def _global_center_delta(self, idx):
        idx = int(idx)
        canonical_center = self.mask_centers[self.canonical_frame_idx].to(self.device)
        return self.mask_centers[idx].to(self.device) - canonical_center

    def _camera_info(self):
        canonical_xyz = self.renderer.gaussians.get_xyz.detach()
        canonical_center = canonical_xyz.mean(dim=0, keepdim=True)
        ones = torch.ones((1, 1), dtype=canonical_center.dtype, device=canonical_center.device)
        view_center = torch.cat([canonical_center, ones], dim=-1) @ self.fixed_cam.world_view_transform.to(dtype=canonical_center.dtype, device=canonical_center.device)
        depth_scale = view_center[:, 2].abs().clamp_min(1e-6).reshape(()).detach()
        return {
            "right": self.fixed_cam.right,
            "up": self.fixed_cam.up,
            "tan_half_fovx": float(np.tan(self.fixed_cam.FoVx * 0.5)),
            "tan_half_fovy": float(np.tan(self.fixed_cam.FoVy * 0.5)),
            "depth_scale": depth_scale,
            "canonical_center": canonical_center,
        }

    @torch.no_grad()
    def _align_canonical_template_to_frame_center(self):
        self.fixed_cam.time = float(self.canonical_time)
        out = self.renderer.render(
            self.fixed_cam,
            stage="coarse",
        )
        pred_alpha = out["alpha"].unsqueeze(0)
        pred_center, pred_area = compute_mask_motion_stats(pred_alpha)
        target_center = self.mask_centers[self.canonical_frame_idx].to(self.device)
        center_delta_2d = (target_center - pred_center).reshape(1, 2)

        self.canonical_alignment_offset_2d = center_delta_2d.reshape(-1).detach()
        if float(center_delta_2d.abs().max().item()) < 1e-8:
            self.canonical_alignment_offset_world = torch.zeros(3, dtype=target_center.dtype, device=self.device)
            return

        canonical_xyz = self.renderer.gaussians.get_xyz.detach()
        shift_world = self.renderer.gaussians._deformation.project_image_shift_to_world(
            center_delta_2d,
            canonical_xyz,
            self._camera_info(),
        ).reshape(1, 3)

        self.renderer.gaussians._xyz.add_(shift_world.expand_as(self.renderer.gaussians._xyz))
        self.canonical_alignment_offset_world = shift_world.reshape(-1).detach()
        self._log(
            f"[canonical_align] frame={int(self.canonical_frame_idx)} "
            f"pred_center=({float(pred_center[0]):.4f},{float(pred_center[1]):.4f}) "
            f"target_center=({float(target_center[0]):.4f},{float(target_center[1]):.4f}) "
            f"offset_2d=({float(self.canonical_alignment_offset_2d[0]):.4f},{float(self.canonical_alignment_offset_2d[1]):.4f}) "
            f"offset_world=({float(self.canonical_alignment_offset_world[0]):.4f},{float(self.canonical_alignment_offset_world[1]):.4f},{float(self.canonical_alignment_offset_world[2]):.4f}) "
            f"pred_area={float(pred_area):.6f}"
        )

    def _save_segment_debug_json(self):
        payload = {
            "step": int(self.step),
            "stage": self.current_train_stage,
            "canonical_frame_idx": int(self.canonical_frame_idx),
            "segments": [],
        }
        for seg in self.motion_segments:
            segment_id = int(seg["segment_id"])
            traj = self.renderer.gaussians._deformation.get_segment_translations(segment_id).detach().cpu().tolist()
            gt_center_2d = [self.mask_centers[idx].detach().cpu().tolist() for idx in seg["indices"]]
            gt_delta_2d = [self._global_center_delta(idx).detach().cpu().tolist() for idx in seg["indices"]]
            traj_error_2d = [
                (self.renderer.gaussians._deformation.get_segment_translation(idx).detach().cpu().reshape(-1)[:2] - self._global_center_delta(idx).detach().cpu()).tolist()
                for idx in seg["indices"]
            ]
            bridge_2d = None
            left_velocity_2d = None
            right_velocity_2d = None
            left_ode_velocity_2d = None
            right_ode_velocity_2d = None
            if seg["segment_type"] == "hidden" and not (self.current_train_stage == "TRAJ" and self.step < self.traj_hidden_start_step):
                use_gt_boundary = False
                bridge_2d = self._hidden_bridge_targets(seg, use_gt_boundary=use_gt_boundary, detach_boundary=True).detach().cpu().tolist()
                left_velocity_2d, right_velocity_2d = self._estimate_boundary_states(seg, use_gt_boundary=use_gt_boundary, detach_boundary=True)[2:4]
                left_ode_velocity_2d, right_ode_velocity_2d = self._query_boundary_ode_velocity(seg)
                left_velocity_2d = left_velocity_2d.detach().cpu().tolist() if left_velocity_2d is not None else None
                right_velocity_2d = right_velocity_2d.detach().cpu().tolist() if right_velocity_2d is not None else None
                left_ode_velocity_2d = left_ode_velocity_2d.detach().cpu().tolist() if left_ode_velocity_2d is not None else None
                right_ode_velocity_2d = right_ode_velocity_2d.detach().cpu().tolist() if right_ode_velocity_2d is not None else None
            payload["segments"].append(
                {
                    "segment_id": segment_id,
                    "segment_type": seg["segment_type"],
                    "start_idx": int(seg["start_idx"]),
                    "end_idx": int(seg["end_idx"]),
                    "ref_idx": int(seg["ref_idx"]),
                    "left_anchor_idx": seg["left_anchor_idx"],
                    "right_anchor_idx": seg["right_anchor_idx"],
                    "local_times": [float(v) for v in seg["local_times"]],
                    "trajectory_2d": traj,
                    "gt_center_2d": gt_center_2d,
                    "gt_center_delta_to_ref": gt_delta_2d,
                    "traj_center_error": traj_error_2d,
                    "hidden_bridge_2d": bridge_2d,
                    "left_boundary_velocity_2d": left_velocity_2d,
                    "right_boundary_velocity_2d": right_velocity_2d,
                    "left_ode_velocity_2d": left_ode_velocity_2d,
                    "right_ode_velocity_2d": right_ode_velocity_2d,
                }
            )
        out_path = os.path.join(self.log_dir, f"segment_debug_{self.step:05d}.json")
        with open(out_path, "w", encoding="ascii") as f:
            json.dump(payload, f, indent=2)

    def _traj_pred_center(self, idx):
        canonical_center = self.mask_centers[self.canonical_frame_idx].to(self.device)
        return canonical_center + self._segment_translation(idx)[:2]

    def _traj_bridge_center_map(self):
        bridge_centers = {}
        canonical_center = self.mask_centers[self.canonical_frame_idx].detach().cpu()
        for seg in self.motion_segments:
            if seg["segment_type"] != "hidden":
                continue
            bridge = self._hidden_bridge_targets(seg, use_gt_boundary=False, detach_boundary=True).detach().cpu()
            for local_idx, frame_idx in enumerate(seg["indices"]):
                bridge_centers[int(frame_idx)] = canonical_center + bridge[local_idx]
        return bridge_centers

    def _draw_polyline(self, canvas, points, color, thickness=2):
        h, w = canvas.shape[:2]
        polyline = []
        for point in points:
            x = int(np.clip(round(float(point[0]) * (w - 1)), 0, w - 1))
            y = int(np.clip(round(float(point[1]) * (h - 1)), 0, h - 1))
            polyline.append((x, y))
        if len(polyline) >= 2:
            cv2.polylines(canvas, [np.asarray(polyline, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def _make_traj_vis_cam(self, time_value):
        pose = self.fixed_cam.c2w.detach().cpu().numpy()
        fov_scale = max(1.0, float(self.traj_render_fov_scale))
        fovy = min(float(self.fixed_cam.FoVy) * fov_scale, np.deg2rad(170.0))
        width = int(max(64, self.traj_render_size))
        height = int(max(64, self.traj_render_size))
        aspect = float(width) / float(height)
        fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)
        return MiniCam(
            pose,
            width,
            height,
            fovy,
            fovx,
            self.fixed_cam.znear,
            self.fixed_cam.zfar,
            time=float(time_value),
        )

    def _make_ode_diag_cam(self, time_value):
        pose = self.fixed_cam.c2w.detach().cpu().numpy()
        fov_scale = max(1.0, float(self.ode_diag_fov_scale))
        fovy = min(float(self.fixed_cam.FoVy) * fov_scale, np.deg2rad(170.0))
        width = int(max(64, self.ode_diag_size))
        height = int(max(64, self.ode_diag_size))
        aspect = float(width) / float(height)
        fovx = 2.0 * np.arctan(np.tan(fovy * 0.5) * aspect)
        return MiniCam(
            pose,
            width,
            height,
            fovy,
            fovx,
            self.fixed_cam.znear,
            self.fixed_cam.zfar,
            time=float(time_value),
        )

    def _draw_original_fov_boundary(self, image, wide_cam):
        if image is None:
            return
        orig_ratio_x = np.tan(float(self.fixed_cam.FoVx) * 0.5) / max(np.tan(float(wide_cam.FoVx) * 0.5), 1e-8)
        orig_ratio_y = np.tan(float(self.fixed_cam.FoVy) * 0.5) / max(np.tan(float(wide_cam.FoVy) * 0.5), 1e-8)
        h, w = image.shape[:2]
        box_w = int(np.clip(round(w * orig_ratio_x), 1, w))
        box_h = int(np.clip(round(h * orig_ratio_y), 1, h))
        x0 = max(0, (w - box_w) // 2)
        y0 = max(0, (h - box_h) // 2)
        x1 = min(w - 1, x0 + box_w - 1)
        y1 = min(h - 1, y0 + box_h - 1)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(image, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (0, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(image, "original FOV", (x0 + 8, max(22, y0 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    def _orig_to_wide_norm(self, center, wide_cam):
        orig_ratio_x = np.tan(float(self.fixed_cam.FoVx) * 0.5) / max(np.tan(float(wide_cam.FoVx) * 0.5), 1e-8)
        orig_ratio_y = np.tan(float(self.fixed_cam.FoVy) * 0.5) / max(np.tan(float(wide_cam.FoVy) * 0.5), 1e-8)
        x = 0.5 + (float(center[0]) - 0.5) * orig_ratio_x
        y = 0.5 + (float(center[1]) - 0.5) * orig_ratio_y
        return np.array([x, y], dtype=np.float32)

    def _project_world_points_to_image(self, points, cam):
        if points.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
        pts = points.detach()
        ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
        pts_h = torch.cat([pts, ones], dim=-1)
        full_proj = cam.full_proj_transform.to(dtype=pts.dtype, device=pts.device)
        clip = pts_h @ full_proj
        clip_w = clip[:, 3:4]
        safe_w = torch.where(clip_w.abs() > 1e-6, clip_w, torch.full_like(clip_w, 1e-6))
        ndc = clip[:, :3] / safe_w
        norm_x = 0.5 + 0.5 * ndc[:, 0]
        norm_y = 0.5 - 0.5 * ndc[:, 1]
        px = norm_x * float(cam.image_width - 1)
        py = norm_y * float(cam.image_height - 1)
        coords = torch.stack([px, py], dim=-1)
        valid = (clip_w.squeeze(-1) > 0) & (norm_x >= 0.0) & (norm_x <= 1.0) & (norm_y >= 0.0) & (norm_y <= 1.0)
        return coords.detach().cpu().numpy().astype(np.float32), valid.detach().cpu().numpy()

    def _draw_delta_heatmap(self, canvas, cam, traj_xyz, ode_xyz):
        delta = torch.norm(ode_xyz - traj_xyz, dim=-1)
        mean_delta = float(delta.mean().item()) if delta.numel() > 0 else 0.0
        max_delta = float(delta.max().item()) if delta.numel() > 0 else 0.0
        coords, valid = self._project_world_points_to_image(ode_xyz, cam)
        if delta.numel() == 0 or not np.any(valid):
            return mean_delta, max_delta

        delta_np = delta.detach().cpu().numpy()
        valid_idx = np.where(valid)[0]
        if valid_idx.shape[0] == 0:
            return mean_delta, max_delta

        draw_cap = max(1, int(self.ode_delta_draw_points))
        if valid_idx.shape[0] > draw_cap:
            order = np.argsort(delta_np[valid_idx])[::-1][:draw_cap]
            valid_idx = valid_idx[order]

        delta_draw = delta_np[valid_idx]
        scale = max(float(np.percentile(delta_draw, 95)) if delta_draw.size > 0 else 0.0, 1e-8)
        norm_vals = np.clip(delta_draw / scale, 0.0, 1.0)
        color_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)[:, 0, :]
        for point_idx, norm_val in zip(valid_idx, norm_vals):
            x = int(np.clip(round(float(coords[point_idx, 0])), 0, canvas.shape[1] - 1))
            y = int(np.clip(round(float(coords[point_idx, 1])), 0, canvas.shape[0] - 1))
            color = tuple(int(v) for v in color_lut[int(round(norm_val * 255.0))])
            cv2.circle(canvas, (x, y), 1, color, -1, cv2.LINE_AA)
        return mean_delta, max_delta

    @torch.no_grad()
    def save_ode_residual_diagnostics(self, name="front_ode_residual_diag"):
        out_dir = os.path.join("valid", self.opt.save_path, f"{self.step}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        video_frames = []
        canonical_xyz = self.renderer.gaussians.get_xyz.detach()

        for idx in range(self.frames):
            time_value = float(idx / self.time_denom)
            self.fixed_cam.time = time_value
            diag_cam = self._make_ode_diag_cam(time_value)

            input_frame = self.input_img_torch_batch[idx][0].detach().permute(1, 2, 0).cpu().numpy()
            input_frame = np.clip(input_frame * 255.0, 0, 255).astype(np.uint8)
            gt_panel = cv2.cvtColor(input_frame, cv2.COLOR_RGB2BGR)

            traj_out = self.renderer.render(
                diag_cam,
                stage="fine",
                vis_ratio=self.mask_vis_ratios[idx],
                motion_mode="traj",
                frame_idx=idx,
            )
            ode_out = self.renderer.render(
                diag_cam,
                stage="fine",
                vis_ratio=self.mask_vis_ratios[idx],
                motion_mode="ode",
                frame_idx=idx,
            )
            residual_delta = ode_out["xyz"] - traj_out["xyz"]
            residual_xyz = canonical_xyz + residual_delta
            residual_out = self.renderer.render_custom_gaussians(
                diag_cam,
                residual_xyz,
                traj_out["scales"],
                traj_out["rot"],
                traj_out["opacity"],
                shs=traj_out["color"],
            )

            traj_panel = cv2.cvtColor(
                np.clip(traj_out["image"].detach().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            residual_panel = cv2.cvtColor(
                np.clip(residual_out["image"].detach().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            heatmap_panel = np.full_like(traj_panel, 12)

            self._draw_original_fov_boundary(traj_panel, diag_cam)
            self._draw_original_fov_boundary(residual_panel, diag_cam)
            self._draw_original_fov_boundary(heatmap_panel, diag_cam)

            mean_delta, max_delta = self._draw_delta_heatmap(heatmap_panel, diag_cam, canonical_xyz, residual_xyz)

            gt_center = self.mask_centers[idx].detach().cpu().numpy()
            gt_center_diag = self._orig_to_wide_norm(gt_center, diag_cam)
            px = int(np.clip(round(float(gt_center_diag[0]) * (diag_cam.image_width - 1)), 0, diag_cam.image_width - 1))
            py = int(np.clip(round(float(gt_center_diag[1]) * (diag_cam.image_height - 1)), 0, diag_cam.image_height - 1))
            for panel in (traj_panel, residual_panel, heatmap_panel):
                cv2.circle(panel, (px, py), 5, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(gt_panel, f"GT frame={idx:03d}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(traj_panel, f"TRAJ wide ref  fov={self.ode_diag_fov_scale:.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(residual_panel, f"PURE ODE residual render  fov={self.ode_diag_fov_scale:.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(heatmap_panel, f"Residual mean={mean_delta:.5f} max={max_delta:.5f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(heatmap_panel, "residual = canonical + (xyz_ode - xyz_traj)", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

            top = np.concatenate([gt_panel, traj_panel], axis=1)
            bottom = np.concatenate([residual_panel, heatmap_panel], axis=1)
            combined = np.concatenate([top, bottom], axis=0)
            cv2.imwrite(os.path.join(out_dir, f"{idx:03d}.png"), combined)
            video_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        imageio.mimwrite(os.path.join(out_dir, "video.mp4"), video_frames, fps=10)

    @torch.no_grad()
    def save_traj_visualization(self):
        out_dir = os.path.join(self.traj_vis_root, f"step_{self.step:05d}")
        os.makedirs(out_dir, exist_ok=True)

        pred_centers = [self._traj_pred_center(i).detach().cpu().numpy() for i in range(self.frames)]
        gt_centers = [self.mask_centers[i].detach().cpu().numpy() for i in range(self.frames)]
        bridge_centers = self._traj_bridge_center_map()
        segment_pred_points = []
        for seg in self.motion_segments:
            seg_id = int(seg["segment_id"])
            seg_type = seg["segment_type"]
            points = [pred_centers[int(frame_idx)] for frame_idx in seg["indices"]]
            segment_pred_points.append((seg_id, seg_type, points))

        video_frames = []
        seg_colors = {
            "visible": (40, 90, 240),
            "boundary": (40, 180, 250),
            "hidden": (180, 70, 230),
        }
        gt_color = (60, 200, 60)
        pred_color = (20, 20, 220)
        bridge_color = (255, 140, 0)
        text_color = (40, 40, 40)

        for idx in range(self.frames):
            frame = self.input_img_torch_batch[idx][0].detach().permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            left = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            right = np.full_like(left, 255)
            traj_cam = self._make_traj_vis_cam(float(idx / self.time_denom))
            traj_out = self.renderer.render(
                traj_cam,
                stage="fine",
                vis_ratio=self.mask_vis_ratios[idx],
                motion_mode="traj",
                frame_idx=idx,
            )
            traj_render = traj_out["image"].detach().permute(1, 2, 0).cpu().numpy()
            traj_render = np.clip(traj_render * 255.0, 0, 255).astype(np.uint8)
            traj_render = cv2.cvtColor(traj_render, cv2.COLOR_RGB2BGR)
            if traj_render.shape[:2] != left.shape[:2]:
                traj_render = cv2.resize(traj_render, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            self._draw_original_fov_boundary(traj_render, traj_cam)

            wide_segment_points = []
            for seg_id, seg_type, points in segment_pred_points:
                wide_points = [self._orig_to_wide_norm(point, traj_cam) for point in points]
                wide_segment_points.append((seg_id, seg_type, wide_points))
            wide_gt_centers = [self._orig_to_wide_norm(center, traj_cam) for center in gt_centers]
            wide_bridge_points = [self._orig_to_wide_norm(bridge_centers[i].numpy(), traj_cam) for i in sorted(bridge_centers.keys())]

            for _, seg_type, points in segment_pred_points:
                self._draw_polyline(right, points, seg_colors[seg_type], thickness=2)
            self._draw_polyline(right, gt_centers, gt_color, thickness=1)
            hidden_bridge_points = [bridge_centers[i].numpy() for i in sorted(bridge_centers.keys())]
            if len(hidden_bridge_points) >= 2:
                self._draw_polyline(right, hidden_bridge_points, bridge_color, thickness=2)
            for _, seg_type, points in wide_segment_points:
                self._draw_polyline(traj_render, points, seg_colors[seg_type], thickness=2)
            self._draw_polyline(traj_render, wide_gt_centers, gt_color, thickness=1)
            if len(wide_bridge_points) >= 2:
                self._draw_polyline(traj_render, wide_bridge_points, bridge_color, thickness=2)

            pred_center = pred_centers[idx]
            gt_center = gt_centers[idx]

            def _to_px(center, image):
                h, w = image.shape[:2]
                return (
                    int(np.clip(round(float(center[0]) * (w - 1)), 0, w - 1)),
                    int(np.clip(round(float(center[1]) * (h - 1)), 0, h - 1)),
                )

            pred_px_left = _to_px(pred_center, left)
            gt_px_left = _to_px(gt_center, left)
            pred_px_right = _to_px(pred_center, right)
            gt_px_right = _to_px(gt_center, right)
            pred_px_traj = _to_px(self._orig_to_wide_norm(pred_center, traj_cam), traj_render)
            gt_px_traj = _to_px(self._orig_to_wide_norm(gt_center, traj_cam), traj_render)

            cv2.circle(left, gt_px_left, 6, gt_color, 2, cv2.LINE_AA)
            cv2.circle(left, pred_px_left, 5, pred_color, -1, cv2.LINE_AA)
            cv2.circle(right, gt_px_right, 6, gt_color, 2, cv2.LINE_AA)
            cv2.circle(right, pred_px_right, 5, pred_color, -1, cv2.LINE_AA)
            cv2.circle(traj_render, gt_px_traj, 6, gt_color, 2, cv2.LINE_AA)
            cv2.circle(traj_render, pred_px_traj, 5, pred_color, -1, cv2.LINE_AA)

            if idx in bridge_centers:
                bridge_px_left = _to_px(bridge_centers[idx].numpy(), left)
                bridge_px_right = _to_px(bridge_centers[idx].numpy(), right)
                bridge_px_traj = _to_px(self._orig_to_wide_norm(bridge_centers[idx].numpy(), traj_cam), traj_render)
                cv2.circle(left, bridge_px_left, 4, bridge_color, 2, cv2.LINE_AA)
                cv2.circle(right, bridge_px_right, 4, bridge_color, 2, cv2.LINE_AA)
                cv2.circle(traj_render, bridge_px_traj, 4, bridge_color, 2, cv2.LINE_AA)

            seg_id = int(self.frame_to_segment_id[idx])
            seg_type = self.motion_segments_by_id[seg_id]["segment_type"]
            cv2.putText(left, f"step={self.step:05d} frame={idx:03d} seg={seg_type}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, "TRAJ map: per-segment lines | green=GT red=pred orange=hidden_bridge", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 2, cv2.LINE_AA)
            cv2.putText(traj_render, f"TRAJ-only render  fov_scale={self.traj_render_fov_scale:.2f}  green=GT red=pred orange=bridge", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            combined = np.concatenate([left, right, traj_render], axis=1)
            cv2.imwrite(os.path.join(out_dir, f"{idx:03d}.png"), combined)
            video_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        imageio.mimwrite(os.path.join(out_dir, "traj_video.mp4"), video_frames, fps=10)

    def _segment_transition_loss(self, segment, allow_left=True, allow_right=True, detach_left=False, detach_right=False):
        losses = []
        if allow_left and segment["left_anchor_idx"] is not None:
            left_t = self._segment_translation(segment["left_anchor_idx"])[:2]
            if detach_left:
                left_t = left_t.detach()
            cur_t = self._segment_translation(segment["start_idx"])[:2]
            losses.append(F.mse_loss(cur_t, left_t))
        if allow_right and segment["right_anchor_idx"] is not None:
            right_t = self._segment_translation(segment["right_anchor_idx"])[:2]
            if detach_right:
                right_t = right_t.detach()
            cur_t = self._segment_translation(segment["end_idx"])[:2]
            losses.append(F.mse_loss(cur_t, right_t))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _trajectory_smoothness_loss(self, segment):
        smooth_indices = list(segment["indices"])
        if segment["left_anchor_idx"] is not None:
            smooth_indices = [int(segment["left_anchor_idx"])] + smooth_indices
        if segment["right_anchor_idx"] is not None:
            smooth_indices = smooth_indices + [int(segment["right_anchor_idx"])]
        if len(smooth_indices) < 3:
            return torch.tensor(0.0, device=self.device)
        translations = [self._segment_translation(idx) for idx in smooth_indices]
        vals = []
        for i in range(1, len(translations) - 1):
            idx_triplet = smooth_indices[i - 1:i + 2]
            weight = self._smoothness_triplet_weight(idx_triplet)
            vals.append(weight * ((translations[i + 1] - 2.0 * translations[i] + translations[i - 1]) ** 2).mean())
        return torch.stack(vals).mean()

    def _trajectory_boundary_loss(self, segment, allow_left=True, allow_right=True, detach_left=False, detach_right=False):
        losses = []
        if allow_left and segment["left_anchor_idx"] is not None:
            losses.append(self._boundary_alignment_loss(segment["left_anchor_idx"], segment["start_idx"], reverse=False, detach_left=detach_left, detach_right=False))
        if allow_right and segment["right_anchor_idx"] is not None:
            losses.append(self._boundary_alignment_loss(segment["end_idx"], segment["right_anchor_idx"], reverse=False, detach_left=False, detach_right=detach_right))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _trajectory_center_loss(self, segment, frame_idx=None):
        losses = []
        indices = [int(frame_idx)] if frame_idx is not None else segment["indices"]
        for idx in indices:
            pred_delta = self._segment_translation(idx)[:2]
            target_delta = self._global_center_delta(idx)
            losses.append(F.mse_loss(pred_delta, target_delta))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _boundary_observation_confidence(self, idx):
        area, border, _ = self._frame_observation_stats(idx)
        return max(self.boundary_observation_floor, area * (1.0 - border))

    def _is_reliable_boundary_frame(self, idx):
        idx = int(idx)
        area, border, _ = self._frame_observation_stats(idx)
        conf = self._boundary_observation_confidence(idx)
        return conf >= self.traj_boundary_reliable_threshold and area >= self.traj_boundary_min_area and border <= self.traj_boundary_max_border

    def _is_reliable_boundary_segment(self, segment):
        if segment["segment_type"] != "boundary":
            return False
        return any(self._is_reliable_boundary_frame(idx) for idx in segment["indices"])

    def _segment_edge_visibility(self, segment):
        left_visible = segment["left_anchor_idx"] is not None and self.frame_states[int(segment["left_anchor_idx"])] == "visible"
        right_visible = segment["right_anchor_idx"] is not None and self.frame_states[int(segment["right_anchor_idx"])] == "visible"
        return left_visible, right_visible

    def _find_visible_neighbor(self, idx, direction):
        idx = int(idx)
        direction = 1 if direction >= 0 else -1
        j = idx + direction
        while 0 <= j < self.frames:
            if self.frame_states[j] == "visible":
                return j
            j += direction
        return None

    def _boundary_visible_anchor_loss(self, segment):
        if segment["segment_type"] != "boundary":
            return torch.tensor(0.0, device=self.device)
        losses = []
        left_visible = self._find_visible_neighbor(segment["start_idx"], -1)
        right_visible = self._find_visible_neighbor(segment["end_idx"], +1)
        if left_visible is not None:
            left_t = self._segment_translation(left_visible)[:2].detach()
            cur_t = self._segment_translation(segment["start_idx"])[:2]
            losses.append(F.mse_loss(cur_t, left_t))
        if right_visible is not None:
            right_t = self._segment_translation(right_visible)[:2].detach()
            cur_t = self._segment_translation(segment["end_idx"])[:2]
            losses.append(F.mse_loss(cur_t, right_t))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _trajectory_full_segment_center_loss(self, segment):
        losses = []
        seg_type = segment["segment_type"]
        seg_weight = self.traj_center_factor
        if seg_type == "boundary":
            seg_weight = self.traj_center_factor * max(1.0, self.boundary_center_factor)
        for idx in segment["indices"]:
            idx = int(idx)
            pred_delta = self._segment_translation(idx)[:2]
            target_delta = self._global_center_delta(idx)
            frame_weight = float(np.clip(self.frame_rank_scores[idx], 0.1, 1.0))
            motion_weight = float(self.mask_center_motion_weights[idx])
            losses.append(seg_weight * frame_weight * motion_weight * F.mse_loss(pred_delta, target_delta))
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _shift_canonical_mask_2d(self, delta_2d):
        canonical_mask = self.canonical_mask_template
        dtype = canonical_mask.dtype
        device = canonical_mask.device
        _, _, h, w = canonical_mask.shape
        shift = delta_2d.to(device=device, dtype=dtype)
        ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        sample_x = 2.0 * (grid_x - shift[0]) - 1.0
        sample_y = 2.0 * (grid_y - shift[1]) - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
        return F.grid_sample(
            canonical_mask,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _trajectory_boundary_area_loss(self, segment):
        if segment["segment_type"] != "boundary":
            return torch.tensor(0.0, device=self.device)
        losses = []
        for idx in segment["indices"]:
            idx = int(idx)
            pred_delta = self._segment_translation(idx)[:2]
            pred_mask = self._shift_canonical_mask_2d(pred_delta)
            pred_center, pred_area = compute_mask_motion_stats(pred_mask)
            target_area = self.mask_areas[idx].to(self.device)
            area_err = F.mse_loss(pred_area, target_area)
            frame_weight = float(np.clip(self.frame_rank_scores[idx], 0.1, 1.0))
            weighted_area_err = self.traj_boundary_area_factor * frame_weight * area_err
            losses.append(weighted_area_err)
            if hasattr(self, "_boundary_area_debug_records"):
                side_touches = self.frame_border_side_touches[idx]
                dominant_side = max(side_touches, key=side_touches.get)
                self._boundary_area_debug_records.append(
                    {
                        "frame_idx": int(idx),
                        "pred_center_x": float(pred_center[0].detach().item()),
                        "pred_center_y": float(pred_center[1].detach().item()),
                        "pred_area": float(pred_area.detach().item()),
                        "gt_area": float(target_area.detach().item()),
                        "area_loss": float(weighted_area_err.detach().item()),
                        "dominant_side": dominant_side,
                        "side_touches": {k: float(v) for k, v in side_touches.items()},
                    }
                )
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _trajectory_boundary_one_sided_loss(self, segment):
        if segment["segment_type"] != "boundary":
            return torch.tensor(0.0, device=self.device)
        losses = []
        for idx in segment["indices"]:
            side_touches = self.frame_border_side_touches[int(idx)]
            pred_delta = self._segment_translation(idx)[:2]
            gt_delta = self._global_center_delta(idx)
            side_losses = []
            if side_touches["right"] > 0.0:
                side_losses.append(side_touches["right"] * torch.relu(gt_delta[0] - pred_delta[0]) ** 2)
            if side_touches["left"] > 0.0:
                side_losses.append(side_touches["left"] * torch.relu(pred_delta[0] - gt_delta[0]) ** 2)
            if side_touches["bottom"] > 0.0:
                side_losses.append(side_touches["bottom"] * torch.relu(gt_delta[1] - pred_delta[1]) ** 2)
            if side_touches["top"] > 0.0:
                side_losses.append(side_touches["top"] * torch.relu(pred_delta[1] - gt_delta[1]) ** 2)
            if len(side_losses) == 0:
                continue
            frame_weight = float(np.clip(self.frame_rank_scores[idx], 0.1, 1.0))
            losses.append(self.traj_boundary_one_sided_factor * frame_weight * torch.stack(side_losses).sum())
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _trajectory_energy_loss(self, segment):
        indices = list(segment["indices"])
        if segment["left_anchor_idx"] is not None:
            indices = [int(segment["left_anchor_idx"])] + indices
        if segment["right_anchor_idx"] is not None:
            indices = indices + [int(segment["right_anchor_idx"])]
        if len(indices) < 2:
            return torch.tensor(0.0, device=self.device)
        translations = torch.stack([self._segment_translation(idx)[:2] for idx in indices], dim=0)
        deltas = translations[1:] - translations[:-1]
        return (deltas ** 2).mean()

    def _global_trajectory_smoothness_loss(self):
        if self.frames < 3:
            return torch.tensor(0.0, device=self.device)
        translations = torch.stack([self._segment_translation(idx)[:2] for idx in range(self.frames)], dim=0)
        vals = []
        for idx in range(1, self.frames - 1):
            triplet = [idx - 1, idx, idx + 1]
            weight = self._smoothness_triplet_weight(triplet)
            vals.append(weight * ((translations[idx + 1] - 2.0 * translations[idx] + translations[idx - 1]) ** 2).mean())
        if len(vals) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(vals).mean()

    def _find_non_hidden_neighbor(self, idx, direction):
        idx = int(idx)
        direction = 1 if direction >= 0 else -1
        j = idx + direction
        while 0 <= j < self.frames:
            if j not in self.hidden_frame_set:
                return j
            j += direction
        return None

    def _finite_difference_velocity_2d(self, idx_a, idx_b):
        if idx_a is None or idx_b is None:
            return None
        ta = float(idx_a / self.time_denom)
        tb = float(idx_b / self.time_denom)
        dt = tb - ta
        if abs(dt) < 1e-8:
            return None
        return (self._segment_translation(idx_b)[:2] - self._segment_translation(idx_a)[:2]) / dt

    def _center_delta(self, idx, ref_idx):
        return self.mask_centers[int(idx)].to(self.device) - self.mask_centers[int(ref_idx)].to(self.device)

    def _finite_difference_center_velocity_2d(self, idx_a, idx_b, ref_idx):
        if idx_a is None or idx_b is None:
            return None
        ta = float(idx_a / self.time_denom)
        tb = float(idx_b / self.time_denom)
        dt = tb - ta
        if abs(dt) < 1e-8:
            return None
        return (self._center_delta(idx_b, ref_idx) - self._center_delta(idx_a, ref_idx)) / dt

    def _estimate_boundary_states(self, segment, use_gt_boundary=False, detach_boundary=False):
        left_idx = segment["left_anchor_idx"]
        right_idx = segment["right_anchor_idx"]
        if left_idx is None or right_idx is None:
            return None, None, None, None, None, None
        left_prev = self._find_non_hidden_neighbor(left_idx, -1)
        right_next = self._find_non_hidden_neighbor(right_idx, +1)
        if use_gt_boundary:
            ref_idx = self.canonical_frame_idx
            p_left = self._global_center_delta(left_idx)
            p_right = self._global_center_delta(right_idx)
            v_left = self._finite_difference_center_velocity_2d(left_prev, left_idx, ref_idx)
            v_right = self._finite_difference_center_velocity_2d(right_idx, right_next, ref_idx)
        else:
            p_left = self._segment_translation(left_idx)[:2]
            p_right = self._segment_translation(right_idx)[:2]
            v_left = self._finite_difference_velocity_2d(left_prev, left_idx)
            v_right = self._finite_difference_velocity_2d(right_idx, right_next)
        if v_left is None:
            v_left = torch.zeros(2, device=self.device, dtype=p_left.dtype)
        if v_right is None:
            v_right = torch.zeros(2, device=self.device, dtype=p_right.dtype)
        if detach_boundary and not use_gt_boundary:
            p_left = p_left.detach()
            p_right = p_right.detach()
            v_left = v_left.detach()
            v_right = v_right.detach()
        t_left = float(left_idx / self.time_denom)
        t_right = float(right_idx / self.time_denom)
        return p_left, p_right, v_left, v_right, t_left, t_right

    def _hermite_bridge_2d(self, p_left, v_left, p_right, v_right, local_times, duration):
        u = torch.as_tensor(local_times, dtype=p_left.dtype, device=self.device).reshape(-1, 1)
        duration = max(float(duration), 1e-6)
        m_left = v_left.reshape(1, 2) * duration
        m_right = v_right.reshape(1, 2) * duration
        p_left = p_left.reshape(1, 2)
        p_right = p_right.reshape(1, 2)
        h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
        h10 = u ** 3 - 2.0 * u ** 2 + u
        h01 = -2.0 * u ** 3 + 3.0 * u ** 2
        h11 = u ** 3 - u ** 2
        return h00 * p_left + h10 * m_left + h01 * p_right + h11 * m_right

    def _hidden_bridge_targets(self, segment, use_gt_boundary=False, detach_boundary=False):
        p_left, p_right, v_left, v_right, t_left, t_right = self._estimate_boundary_states(
            segment,
            use_gt_boundary=use_gt_boundary,
            detach_boundary=detach_boundary,
        )
        if p_left is None or p_right is None:
            return torch.stack([self._segment_translation(idx)[:2] for idx in segment["indices"]], dim=0)
        duration = t_right - t_left
        hermite = self._hermite_bridge_2d(p_left, v_left, p_right, v_right, segment["local_times"], duration)
        u = torch.as_tensor(segment["local_times"], dtype=hermite.dtype, device=self.device).reshape(-1, 1)
        linear = (1.0 - u) * p_left.reshape(1, 2) + u * p_right.reshape(1, 2)
        blend = float(np.clip(self.hidden_bridge_hermite_factor, 0.0, 1.0))
        bridge = blend * hermite + (1.0 - blend) * linear

        exit_side = self._hidden_exit_side(segment)
        exit_margin = float(self.hidden_exit_margin_factor) * float(self.canonical_mask_radius_2d)
        bump = torch.sin(np.pi * u) ** 2
        if exit_side == "right":
            target_peak = 1.0 + exit_margin
            peak = torch.max(bridge[:, 0])
            need = torch.clamp(torch.as_tensor(target_peak, device=self.device, dtype=bridge.dtype) - peak, min=0.0)
            bridge[:, 0:1] = bridge[:, 0:1] + need * bump
        elif exit_side == "left":
            target_peak = -exit_margin
            peak = torch.min(bridge[:, 0])
            need = torch.clamp(peak - torch.as_tensor(target_peak, device=self.device, dtype=bridge.dtype), min=0.0)
            bridge[:, 0:1] = bridge[:, 0:1] - need * bump
        elif exit_side == "bottom":
            target_peak = 1.0 + exit_margin
            peak = torch.max(bridge[:, 1])
            need = torch.clamp(torch.as_tensor(target_peak, device=self.device, dtype=bridge.dtype) - peak, min=0.0)
            bridge[:, 1:2] = bridge[:, 1:2] + need * bump
        elif exit_side == "top":
            target_peak = -exit_margin
            peak = torch.min(bridge[:, 1])
            need = torch.clamp(peak - torch.as_tensor(target_peak, device=self.device, dtype=bridge.dtype), min=0.0)
            bridge[:, 1:2] = bridge[:, 1:2] - need * bump
        return bridge

    def _hidden_bridge_endpoint_velocities(self, segment, use_gt_boundary=False, detach_boundary=False):
        p_left, p_right, v_left, v_right, t_left, t_right = self._estimate_boundary_states(
            segment,
            use_gt_boundary=use_gt_boundary,
            detach_boundary=detach_boundary,
        )
        if p_left is None or p_right is None:
            zero = torch.zeros(2, device=self.device)
            return zero, zero
        return v_left, v_right

    def _hidden_exit_side(self, segment):
        side_scores = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}
        for anchor_idx in (segment["left_anchor_idx"], segment["right_anchor_idx"]):
            if anchor_idx is None:
                continue
            touches = self.frame_border_side_touches[int(anchor_idx)]
            for key in side_scores:
                side_scores[key] += float(touches.get(key, 0.0))
        dominant_side = max(side_scores, key=side_scores.get)
        if side_scores[dominant_side] > 0.0:
            return dominant_side
        p_left, p_right, _, _, _, _ = self._estimate_boundary_states(
            segment,
            use_gt_boundary=False,
            detach_boundary=True,
        )
        if p_left is None or p_right is None:
            return "right"
        if max(float(p_left[0]), float(p_right[0])) > 0.5:
            return "right"
        if min(float(p_left[0]), float(p_right[0])) < -0.5:
            return "left"
        if max(float(p_left[1]), float(p_right[1])) > 0.5:
            return "bottom"
        return "top"

    def _hidden_bridge_loss(self, segment, use_gt_boundary=False, detach_boundary=False):
        if segment["left_anchor_idx"] is None or segment["right_anchor_idx"] is None:
            return torch.tensor(0.0, device=self.device)
        pred = torch.stack([self._segment_translation(idx)[:2] for idx in segment["indices"]], dim=0)
        bridge = self._hidden_bridge_targets(segment, use_gt_boundary=use_gt_boundary, detach_boundary=detach_boundary)
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)
        endpoint_ids = sorted(set([0, pred.shape[0] - 1, pred.shape[0] // 2]))
        endpoint_loss = torch.stack([F.mse_loss(pred[i], bridge[i]) for i in endpoint_ids]).mean()
        if pred.shape[0] > 1:
            trend_loss = F.mse_loss(pred[1:] - pred[:-1], bridge[1:] - bridge[:-1])
        else:
            trend_loss = torch.tensor(0.0, device=self.device)
        return self.hidden_bridge_endpoint_factor * endpoint_loss + self.hidden_bridge_trend_factor * trend_loss

    def _hidden_damping_loss(self, segment, detach_boundary=True):
        if segment["segment_type"] != "hidden":
            return torch.tensor(0.0, device=self.device)
        pred = torch.stack([self._segment_translation(idx)[:2] for idx in segment["indices"]], dim=0)
        if pred.shape[0] < 3:
            return torch.tensor(0.0, device=self.device)
        velocities = pred[1:] - pred[:-1]
        speeds = torch.norm(velocities, dim=-1)
        _, _, v_left, v_right, _, _ = self._estimate_boundary_states(
            segment,
            use_gt_boundary=False,
            detach_boundary=detach_boundary,
        )
        if v_left is None or v_right is None:
            return torch.tensor(0.0, device=self.device)
        left_speed = torch.norm(v_left, p=2)
        right_speed = torch.norm(v_right, p=2)
        min_speed = self.hidden_damping_floor * torch.minimum(left_speed, right_speed)
        num_speeds = speeds.shape[0]
        center_pos = 0.5 * float(num_speeds - 1)
        targets = []
        for i in range(num_speeds):
            if center_pos <= 0.0:
                target = 0.5 * (left_speed + right_speed)
            elif i <= center_pos:
                alpha = float(i / max(center_pos, 1e-6))
                target = (1.0 - alpha) * left_speed + alpha * min_speed
            else:
                alpha = float((i - center_pos) / max((num_speeds - 1) - center_pos, 1e-6))
                target = (1.0 - alpha) * min_speed + alpha * right_speed
            targets.append(target)
        target_speeds = torch.stack(targets)
        envelope_loss = F.mse_loss(speeds, target_speeds)
        if speeds.shape[0] >= 2:
            diffs = speeds[1:] - speeds[:-1]
            mid = max(1, diffs.shape[0] // 2)
            left_mono = torch.relu(diffs[:mid]).mean() if mid > 0 else torch.tensor(0.0, device=self.device)
            right_mono = torch.relu(-diffs[mid:]).mean() if diffs.shape[0] - mid > 0 else torch.tensor(0.0, device=self.device)
            mono_loss = left_mono + right_mono
        else:
            mono_loss = torch.tensor(0.0, device=self.device)
        return self.hidden_damping_factor * (envelope_loss + mono_loss)

    def _frame_observation_stats(self, idx):
        area = float(self.mask_areas[int(idx)].item())
        border = float(self.frame_border_touches[int(idx)])
        vis_ratio = float(self.mask_vis_ratios[int(idx)])
        return area, border, vis_ratio

    def _frame_smoothness_factor(self, idx):
        idx = int(idx)
        state = self.frame_states[idx]
        area, border, vis_ratio = self._frame_observation_stats(idx)
        factor = 1.0
        if state == "boundary":
            factor *= self.boundary_smoothness_factor
        elif state == "hidden":
            factor *= self.hidden_smoothness_factor
        visibility_term = max(float(self.visibility_smoothness_floor), float(np.clip(vis_ratio, 0.0, 1.0)))
        border_term = max(0.05, 1.0 - self.border_smoothness_scale * float(np.clip(border, 0.0, 1.0)))
        area_term = max(0.1, min(1.0, area))
        return factor * visibility_term * border_term * area_term

    def _smoothness_triplet_weight(self, idx_triplet):
        weights = [self._frame_smoothness_factor(idx) for idx in idx_triplet]
        return torch.tensor(min(weights), device=self.device, dtype=torch.float32)

    def _traj_stage_smoothness_weight(self, segment):
        seg_type = segment["segment_type"]
        if seg_type == "visible":
            return 0.0
        if seg_type == "boundary":
            return float(getattr(self.opt, "traj_boundary_smooth_factor", 0.05))
        if seg_type == "hidden":
            return float(getattr(self.opt, "traj_hidden_smooth_factor", 1.0))
        return 1.0

    def _boundary_alignment_loss(self, left_idx, right_idx, reverse=False, detach_left=False, detach_right=False):
        left_idx = int(left_idx)
        right_idx = int(right_idx)
        left_center = self.mask_centers[left_idx].to(self.device)
        right_center = self.mask_centers[right_idx].to(self.device)
        right_t = self._segment_translation(right_idx)[:2]
        left_t = self._segment_translation(left_idx)[:2]
        if detach_left:
            left_t = left_t.detach()
        if detach_right:
            right_t = right_t.detach()
        pred_delta = right_t - left_t
        target_delta = right_center - left_center
        if reverse:
            pred_delta = -pred_delta
            target_delta = -target_delta

        area_left, border_left, _ = self._frame_observation_stats(left_idx)
        area_right, border_right, _ = self._frame_observation_stats(right_idx)
        min_area = min(area_left, area_right)
        max_border = max(border_left, border_right)
        obs_conf = max(self.boundary_observation_floor, min_area * (1.0 - max_border))
        event_boost = 1.0 + self.boundary_event_boost * ((1.0 - min_area) + max_border)

        center_loss = obs_conf * F.mse_loss(pred_delta, target_delta)
        pred_norm = torch.norm(pred_delta, p=2)
        target_norm = torch.norm(target_delta, p=2)
        if float(pred_norm.item()) < 1e-8 or float(target_norm.item()) < 1e-8:
            direction_loss = torch.tensor(0.0, device=self.device)
        else:
            cos = F.cosine_similarity(pred_delta.unsqueeze(0), target_delta.unsqueeze(0), dim=-1).squeeze(0)
            direction_loss = 1.0 - torch.clamp(cos, -1.0, 1.0)
        return event_boost * (center_loss + self.boundary_direction_factor * direction_loss)

    def _project_world_velocity_to_image(self, velocity_world, camera_info):
        if velocity_world is None or camera_info is None:
            return None
        right = camera_info["right"].to(dtype=velocity_world.dtype, device=velocity_world.device).reshape(1, 3)
        up = camera_info["up"].to(dtype=velocity_world.dtype, device=velocity_world.device).reshape(1, 3)
        depth_scale = torch.as_tensor(camera_info["depth_scale"], dtype=velocity_world.dtype, device=velocity_world.device).reshape(1, 1)
        x_scale = 2.0 * depth_scale * float(camera_info["tan_half_fovx"])
        y_scale = 2.0 * depth_scale * float(camera_info["tan_half_fovy"])
        vx = (velocity_world * right).sum(dim=-1, keepdim=True) / torch.clamp(x_scale, min=1e-8)
        vy = -(velocity_world * up).sum(dim=-1, keepdim=True) / torch.clamp(y_scale, min=1e-8)
        return torch.cat([vx, vy], dim=-1)

    def _sample_ode_aux_points(self, points):
        max_points = int(getattr(self.opt, "ode_aux_point_sample", 4096))
        if max_points <= 0 or points.shape[0] <= max_points:
            return points, None
        sample_idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
        return points[sample_idx], sample_idx

    def _query_boundary_ode_velocity(self, segment):
        left_idx = segment["left_anchor_idx"]
        right_idx = segment["right_anchor_idx"]
        if left_idx is None or right_idx is None:
            return None, None
        camera_info = self._camera_info()
        canonical_center = self.renderer.gaussians.get_xyz.detach().mean(dim=0, keepdim=True)

        def _query(frame_idx):
            t_val = float(frame_idx / self.time_denom)
            vel_world = self.renderer.gaussians._deformation.query_velocity(
                canonical_center,
                t_val,
                frame_idx=frame_idx,
                camera_info=camera_info,
            )
            vel_2d = self._project_world_velocity_to_image(vel_world, camera_info)
            return vel_2d.reshape(-1)

        return _query(left_idx), _query(right_idx)

    def _hidden_velocity_boundary_loss(self, segment):
        if segment["left_anchor_idx"] is None or segment["right_anchor_idx"] is None:
            return torch.tensor(0.0, device=self.device)
        target_left, target_right = self._hidden_bridge_endpoint_velocities(segment)
        ode_left, ode_right = self._query_boundary_ode_velocity(segment)
        if ode_left is None or ode_right is None:
            return torch.tensor(0.0, device=self.device)
        return 0.5 * (F.mse_loss(ode_left[:2], target_left) + F.mse_loss(ode_right[:2], target_right))

    def _canonical_translation_loss(self):
        canonical_translation = self._segment_translation(self.canonical_frame_idx)
        return (canonical_translation ** 2).mean()

    def _trajectory_stats(self):
        lr = 0.0
        for group in self.optimizer.param_groups:
            if group.get("name", "") == "trajectory":
                lr = float(group.get("lr", 0.0))
                break
        if getattr(self, "current_train_stage", None) not in {"TRAJ", "JOINT"}:
            zero = torch.tensor(0.0, device=self.device)
            return lr, zero, zero
        params = list(self.renderer.gaussians._deformation.get_trajectory_parameters())
        if len(params) == 0:
            zero = torch.tensor(0.0, device=self.device)
            return lr, zero, zero
        with torch.no_grad():
            grad_sq = torch.tensor(0.0, device=self.device)
            param_sq = torch.tensor(0.0, device=self.device)
            for param in params:
                param_sq = param_sq + (param.detach() ** 2).sum()
                if param.grad is not None:
                    grad_sq = grad_sq + (param.grad.detach() ** 2).sum()
        return lr, torch.sqrt(grad_sq), torch.sqrt(param_sq)

    def _set_stage_trainability(self, stage_name):
        canonical_names = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
        hex_names = {"deformation_hex", "grid"}
        ode_names = {"deformation_ode", "video_encoder"}
        traj_names = {"trajectory"}
        train_canonical = bool(getattr(self.opt, "train_canonical", False))
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            base_lr = group.get("base_lr", group.get("lr", 0.0))
            active = False
            if stage_name == "TRAJ":
                active = name in traj_names
            elif stage_name == "ODE":
                active = name in ode_names
            elif stage_name == "HEX":
                active = name in hex_names
            elif stage_name == "JOINT":
                active = name in traj_names or name in hex_names or name in ode_names or (train_canonical and name in canonical_names)
            group["lr"] = base_lr if active else 0.0
            for param in group.get("params", []):
                param.requires_grad_(active)

    def _save_checkpoint(self, step):
        auto_path = os.path.join(self.opt.outdir, f"{self.opt.save_path}{step}")
        os.makedirs(auto_path, exist_ok=True)
        self.renderer.gaussians.save_ply(os.path.join(auto_path, "model.ply"))
        self.renderer.gaussians.save_deformation(auto_path)

    @torch.no_grad()
    def _log_motion_debug(self):
        if self.frames <= 0:
            return

        canonical_xyz = self.renderer.gaussians.get_xyz.detach()
        mid_idx = self.frames // 2
        sample_specs = [
            ("tcanon", self.canonical_time, float(self.mask_vis_ratios[self.canonical_frame_idx])),
            ("t0", 0.0, float(self.mask_vis_ratios[0])),
            ("tmid", float(mid_idx / self.time_denom), float(self.mask_vis_ratios[mid_idx])),
            ("t1", 1.0, float(self.mask_vis_ratios[-1])),
        ]

        deformed = {}
        translations = {}
        motion_mode = self._resolve_motion_mode(self.current_train_stage)
        sample_frame_indices = {
            "tcanon": self.canonical_frame_idx,
            "t0": 0,
            "tmid": mid_idx,
            "t1": self.frames - 1,
        }
        for name, time_value, vis_ratio in sample_specs:
            translations[name] = self._segment_translation(sample_frame_indices[name]).detach().reshape(-1)
            pts, _ = self.renderer.gaussians._deformation.deform_positions(
                canonical_xyz,
                time_value,
                vis_ratio=vis_ratio,
                return_velocities=False,
                motion_mode=motion_mode,
                frame_idx=sample_frame_indices[name],
                camera_info=self._camera_info(),
            )
            deformed[name] = pts

        delta_0_mid = float((deformed["t0"] - deformed["tmid"]).abs().mean().item())
        delta_mid_1 = float((deformed["tmid"] - deformed["t1"]).abs().mean().item())
        drift_canonical = float((deformed["tcanon"] - canonical_xyz).abs().mean().item())
        drift_0 = float((deformed["t0"] - canonical_xyz).abs().mean().item())
        drift_mid = float((deformed["tmid"] - canonical_xyz).abs().mean().item())
        drift_1 = float((deformed["t1"] - canonical_xyz).abs().mean().item())
        self._log(
            f"[motion_debug] step={self.step:05d} "
            f"mean|x0-xmid|={delta_0_mid:.6f} mean|xmid-x1|={delta_mid_1:.6f} "
            f"driftcanon={drift_canonical:.6f} drift0={drift_0:.6f} driftmid={drift_mid:.6f} drift1={drift_1:.6f} "
            f"Tcanon=({translations['tcanon'][0]:.4f},{translations['tcanon'][1]:.4f}) "
            f"T0=({translations['t0'][0]:.4f},{translations['t0'][1]:.4f}) "
            f"Tmid=({translations['tmid'][0]:.4f},{translations['tmid'][1]:.4f}) "
            f"T1=({translations['t1'][0]:.4f},{translations['t1'][1]:.4f})"
        )
        self._save_segment_debug_json()

    @torch.no_grad()
    def save_renderings(self, name="front", motion_mode_override=None):
        images = []
        out_dir = os.path.join("valid", self.opt.save_path, f"{self.step}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        motion_mode = motion_mode_override or self._resolve_motion_mode(self.current_train_stage)
        for i in range(self.frames):
            self.fixed_cam.time = float(i / self.time_denom)
            out = self.renderer.render(
                self.fixed_cam,
                stage="fine",
                vis_ratio=self.mask_vis_ratios[i],
                motion_mode=motion_mode,
                frame_idx=i,
            )
            image = out["image"].unsqueeze(0)
            images.append(image)
            save_image_to_local(image[0].detach(), os.path.join(out_dir, f"{str(i).zfill(2)}.jpg"))
        samples = torch.cat(images, dim=0)
        vid = (
            (samples.permute(0, 2, 3, 1) * 255)
            .clamp(0, 255)
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        imageio.mimwrite(os.path.join(out_dir, "video.mp4"), vid)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        train_stage = self._resolve_stage_name()
        self.current_train_stage = train_stage
        motion_mode = self._resolve_motion_mode(train_stage)
        self.step += 1
        step_ratio = min(1.0, self.step / max(1, int(self.opt.iters)))
        if train_stage == "TRAJ" and not self.observed_traj_initialized:
            self._initialize_observed_segment_trajectories_from_gt()
            self.observed_traj_initialized = True
        if train_stage == "TRAJ" and self._traj_hidden_enabled():
            self._sync_hidden_segment_trajectories_to_bridge(use_gt_boundary=False)
            self.hidden_traj_initialized = True

        self.renderer.gaussians.update_learning_rate(self.step)
        self._set_stage_trainability(train_stage)
        if train_stage == "TRAJ":
            self._set_traj_segment_trainability()
        need_video_context = train_stage in ("ODE", "JOINT")
        if self.step % int(self.opt.valid_interval) == 0 and train_stage in ("ODE", "HEX", "JOINT"):
            need_video_context = True
        if need_video_context:
            _ = self.renderer.gaussians._deformation.set_video_frames(self.video_frames_tensor)
        if self.motion_debug_interval > 0 and self.step % self.motion_debug_interval == 1:
            self._log_motion_debug()

        total_loss = torch.tensor(0.0, device=self.device)
        image_loss = torch.tensor(0.0, device=self.device)
        alpha_loss = torch.tensor(0.0, device=self.device)
        velocity_loss = torch.tensor(0.0, device=self.device)
        curv_loss = torch.tensor(0.0, device=self.device)
        anchor_loss = torch.tensor(0.0, device=self.device)
        cycle_loss = torch.tensor(0.0, device=self.device)
        scale_loss = torch.tensor(0.0, device=self.device)
        ode_center_loss = torch.tensor(0.0, device=self.device)
        ode_area_loss = torch.tensor(0.0, device=self.device)
        traj_center_loss = torch.tensor(0.0, device=self.device)
        traj_boundary_loss = torch.tensor(0.0, device=self.device)
        traj_smooth_loss = torch.tensor(0.0, device=self.device)
        segment_transition_loss = torch.tensor(0.0, device=self.device)
        hidden_bridge_loss = torch.tensor(0.0, device=self.device)
        hidden_velocity_loss = torch.tensor(0.0, device=self.device)
        hidden_energy_loss = torch.tensor(0.0, device=self.device)
        hidden_damping_loss = torch.tensor(0.0, device=self.device)
        canonical_traj_loss = self.lambda_canonical_traj * self._canonical_translation_loss()

        if self.step % int(self.opt.valid_interval) == 0:
            self.save_renderings(name="front")
            if train_stage in ("ODE", "HEX", "JOINT"):
                self.save_renderings(name="front_traj_only", motion_mode_override="traj")
                self.save_renderings(name="front_ode_only", motion_mode_override="ode")
                self.save_ode_residual_diagnostics(name="front_ode_residual_diag")
        if self.step % int(self.opt.save_interval) == 0:
            self._save_checkpoint(self.step)
        sample_records = []
        self._boundary_area_debug_records = []
        traj_smooth_segments = set()
        hidden_bridge_segments = set()

        if train_stage == "TRAJ":
            visible_phase = self._traj_visible_phase_active()
            boundary_phase = self._traj_boundary_phase_active()
            for segment in self.motion_segments:
                seg_id = int(segment["segment_id"])
                seg_type = segment["segment_type"]
                sample_records.append(
                    f"{int(segment['start_idx'])}-{int(segment['end_idx'])}:{seg_type}:seg{seg_id}"
                )

                if seg_type == "boundary" and boundary_phase:
                    traj_boundary_loss = traj_boundary_loss + self._trajectory_boundary_area_loss(segment)
                    traj_boundary_loss = traj_boundary_loss + self._trajectory_boundary_one_sided_loss(segment)
                    continue

        else:
            for _ in range(int(self.opt.batch_size)):
                t_idx = self._sample_frame_for_stage(train_stage)
                cur_time = float(t_idx / self.time_denom)
                vis_ratio = float(self.mask_vis_ratios[t_idx])
                frame_rank_score = float(self.frame_rank_scores[t_idx])
                frame_state = self.frame_states[t_idx]
                segment = self.motion_segments_by_id[int(self.frame_to_segment_id[t_idx])]
                sample_records.append(
                    f"{t_idx}:{frame_state}:seg{int(segment['segment_id'])}:lt={segment['local_times'][t_idx - int(segment['start_idx'])]:.2f}"
                )
                frame_weight = 1.0 if train_stage == "ODE" else float(np.clip(frame_rank_score, 0.1, 1.0))
                rgb_weight = self.lambda_rgb
                alpha_weight = self.lambda_alpha
                vel_weight = self.lambda_vel
                center_weight = 1.0
                area_weight = 0.5
                if train_stage == "ODE":
                    rgb_weight = self.lambda_rgb * self.ode_rgb_factor
                    alpha_weight = self.lambda_alpha * self.ode_alpha_factor
                    vel_weight = self.lambda_vel * self.ode_vel_factor
                    center_weight = self.ode_center_factor
                    area_weight = self.ode_area_factor
                if frame_state == "boundary":
                    rgb_weight *= self.boundary_rgb_factor
                    alpha_weight *= self.boundary_alpha_factor
                    area_weight *= self.boundary_area_factor
                    center_weight *= self.boundary_center_factor
                elif frame_state == "hidden":
                    if train_stage in ("ODE", "JOINT"):
                        hidden_velocity_loss = hidden_velocity_loss + self._hidden_velocity_boundary_loss(segment)
                    rgb_weight *= self.hidden_rgb_factor
                    alpha_weight *= self.hidden_alpha_factor
                    area_weight *= self.hidden_area_factor
                    center_weight *= self.hidden_center_factor

                self.fixed_cam.time = cur_time
                need_velocity_outputs = bool(train_stage in ("ODE", "JOINT") and self.lambda_vel > 0.0)
                if train_stage == "ODE":
                    need_velocity_outputs = need_velocity_outputs and (self.ode_vel_factor > 0.0)
                out = self.renderer.render(
                    self.fixed_cam,
                    stage="fine",
                    vis_ratio=vis_ratio,
                    motion_mode=motion_mode,
                    frame_idx=t_idx,
                    return_velocities=need_velocity_outputs,
                )

                target_img = self.input_img_torch_batch[t_idx]
                target_mask = self.input_mask_torch_batch[t_idx]
                pred_img = out["image"].unsqueeze(0)
                pred_alpha = out["alpha"].unsqueeze(0)

                image_loss = image_loss + step_ratio * rgb_weight * frame_weight * F.mse_loss(pred_img, target_img)
                alpha_loss = alpha_loss + step_ratio * alpha_weight * F.mse_loss(pred_alpha, target_mask)
                pred_center, pred_area = compute_mask_motion_stats(pred_alpha)
                target_center = self.mask_centers[t_idx].to(self.device)
                target_area = self.mask_areas[t_idx].to(self.device)
                center_motion_weight = float(self.mask_center_motion_weights[t_idx])
                ode_center_loss = ode_center_loss + step_ratio * frame_weight * center_weight * center_motion_weight * F.mse_loss(pred_center, target_center)
                ode_area_loss = ode_area_loss + step_ratio * frame_weight * area_weight * F.mse_loss(pred_area, target_area)
                traj_center_loss = traj_center_loss + frame_weight * center_motion_weight * self._trajectory_center_loss(segment, frame_idx=t_idx)

                canonical_xyz = self.renderer.gaussians.get_xyz
                if train_stage == "HEX" and vis_ratio > 1e-6:
                    anchor_loss = anchor_loss + anchor_shape_loss(out["xyz"], canonical_xyz)

                if train_stage in ("ODE", "JOINT"):
                    velocity_loss = velocity_loss + vel_weight * velocity_reg(out.get("velocities", []))
                    # Only smooth trajectories when supervision is weak, e.g. low-visibility / out-of-FOV segments.
                    if t_idx in self.low_frame_set:
                        aux_canonical_xyz, aux_idx = self._sample_ode_aux_points(canonical_xyz)
                        x_mid = out["xyz"] if aux_idx is None else out["xyz"][aux_idx]
                        t_prev = max(0.0, (t_idx - 1) / self.time_denom)
                        t_next = min(1.0, (t_idx + 1) / self.time_denom)
                        vis_prev = self.mask_vis_ratios[max(0, t_idx - 1)]
                        vis_next = self.mask_vis_ratios[min(self.frames - 1, t_idx + 1)]
                        x_prev, _ = self.renderer.gaussians._deformation.deform_positions(
                            aux_canonical_xyz,
                            t_prev,
                            vis_ratio=vis_prev,
                            return_velocities=False,
                            motion_mode=motion_mode,
                            frame_idx=max(0, t_idx - 1),
                            camera_info=self._camera_info(),
                        )
                        x_next, _ = self.renderer.gaussians._deformation.deform_positions(
                            aux_canonical_xyz,
                            t_next,
                            vis_ratio=vis_next,
                            return_velocities=False,
                            motion_mode=motion_mode,
                            frame_idx=min(self.frames - 1, t_idx + 1),
                            camera_info=self._camera_info(),
                        )
                        curv_loss = curv_loss + curvature_loss(x_prev, x_mid, x_next)

                scales = out["scales"]
                r = self.opt.scale_loss_ratio
                scale_loss = scale_loss + (
                    torch.mean(
                        torch.maximum(
                            torch.max(scales, dim=1).values / (torch.min(scales, dim=1).values + 1e-8),
                            torch.ones_like(torch.max(scales, dim=1).values) * r,
                        )
                    )
                    - r
                ) * scales.shape[0]

        total_loss = total_loss + image_loss + alpha_loss
        total_loss = total_loss + self.opt.lambda_tv * self.renderer.gaussians.compute_regulation(
            self.opt.time_smoothness_weight,
            self.opt.l1_time_planes,
            self.opt.plane_tv_weight,
        )

        if train_stage in ("ODE", "JOINT") and len(self.cycle_pairs) > 0:
            exit_idx, reentry_idx = self.cycle_pairs[np.random.randint(0, len(self.cycle_pairs))]
            t_exit = exit_idx / self.time_denom
            t_reentry = reentry_idx / self.time_denom
            vis_exit = self.mask_vis_ratios[exit_idx]
            vis_reentry = self.mask_vis_ratios[reentry_idx]
            canonical_xyz = self.renderer.gaussians.get_xyz
            aux_canonical_xyz, _ = self._sample_ode_aux_points(canonical_xyz)
            x_exit, _ = self.renderer.gaussians._deformation.deform_positions(
                aux_canonical_xyz,
                t_exit,
                vis_ratio=vis_exit,
                return_velocities=False,
                motion_mode=motion_mode,
                frame_idx=exit_idx,
                camera_info=self._camera_info(),
            )
            x_reentry, _ = self.renderer.gaussians._deformation.deform_positions(
                aux_canonical_xyz,
                t_reentry,
                vis_ratio=vis_reentry,
                return_velocities=False,
                motion_mode=motion_mode,
                frame_idx=reentry_idx,
                camera_info=self._camera_info(),
            )
            cycle_loss = cycle_sparse_loss(x_exit, x_reentry)

        total_loss = total_loss + velocity_loss
        total_loss = total_loss + self.lambda_curv * curv_loss
        total_loss = total_loss + self.lambda_anchor * anchor_loss
        total_loss = total_loss + self.lambda_cycle_sparse * cycle_loss
        total_loss = total_loss + self.lambda_ode_center * ode_center_loss
        total_loss = total_loss + self.lambda_ode_area * ode_area_loss
        total_loss = total_loss + self.lambda_traj_center * traj_center_loss
        total_loss = total_loss + self.lambda_traj_boundary * traj_boundary_loss
        total_loss = total_loss + self.lambda_traj_smooth * traj_smooth_loss
        total_loss = total_loss + self.lambda_segment_transition * segment_transition_loss
        total_loss = total_loss + self.lambda_hidden_bridge * hidden_bridge_loss
        total_loss = total_loss + self.lambda_hidden_velocity * hidden_velocity_loss
        total_loss = total_loss + self.lambda_hidden_energy * hidden_energy_loss
        total_loss = total_loss + self.lambda_hidden_damping * hidden_damping_loss
        total_loss = total_loss + canonical_traj_loss
        total_loss = total_loss + scale_loss

        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            traj_lr, traj_grad_norm, traj_param_norm = self._trajectory_stats()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            traj_lr, traj_grad_norm, traj_param_norm = self._trajectory_stats()

        if train_stage == "TRAJ" and self.traj_vis_interval > 0 and self.step % self.traj_vis_interval == 0:
            self.save_traj_visualization()

        update_start = int(getattr(self.opt, "deformation_update_start_iter", 0))
        update_interval = int(getattr(self.opt, "deformation_update_interval", 0))
        use_deformation_table = bool(getattr(self.opt, "use_deformation_table", False))
        if use_deformation_table and update_interval > 0 and self.step >= update_start and self.step % update_interval == 0:
            self.renderer.gaussians.update_deformation_table(float(self.opt.deformation_threshold))

        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)
        deform_count = int(self.renderer.gaussians._deformation_table.sum().item())
        boundary_area_debug = ""
        if train_stage == "TRAJ" and len(self._boundary_area_debug_records) > 0:
            parts = []
            for rec in self._boundary_area_debug_records:
                parts.append(
                    f"f{rec['frame_idx']}:predA={rec['pred_area']:.4f}/gtA={rec['gt_area']:.4f}/loss={rec['area_loss']:.6f}/side={rec['dominant_side']}"
                )
            boundary_area_debug = " barea=[" + "; ".join(parts) + "]"
        self._log(
            f"stage={train_stage} step={self.step:05d} "
            f"loss={float(total_loss):.4f} img={float(image_loss):.4f} alpha={float(alpha_loss):.4f} "
            f"vel={float(velocity_loss):.4f} curv={float(curv_loss):.4f} anchor={float(anchor_loss):.4f} "
            f"cycle={float(cycle_loss):.4f} center={float(ode_center_loss):.4f} area={float(ode_area_loss):.4f} "
            f"traj_c={float(traj_center_loss):.4f} traj_b={float(traj_boundary_loss):.4f} traj_s={float(traj_smooth_loss):.4f} "
            f"seg_t={float(segment_transition_loss):.4f} hbridge={float(hidden_bridge_loss):.4f} hvel={float(hidden_velocity_loss):.4f} henergy={float(hidden_energy_loss):.4f} hdamp={float(hidden_damping_loss):.4f} ctraj={float(canonical_traj_loss):.4f} "
            f"traj_lr={traj_lr:.6f} traj_g={float(traj_grad_norm):.6f} traj_p={float(traj_param_norm):.6f} "
            f"scale={float(scale_loss):.4f} deform={deform_count} "
            f"time={elapsed:.2f}ms "
            f"samples={'|'.join(sample_records)}"
            f"{boundary_area_debug}"
        )

    def train(self):
        self.prepare_train()
        for _ in tqdm.trange(int(self.opt.iters)):
            self.train_step()
        self._save_checkpoint(self.step)
        self.save_renderings(name="front_final")
        self._log_fh.close()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    trainer = MotionTrainer(opt)
    trainer.train()
