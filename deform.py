import itertools
from typing import Collection, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_integrator import integrate_euler
from velocity_field import VelocityField
from video_encoder import VideoEncoder


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim not in (2, 3):
        raise NotImplementedError(f"Unsupported grid dim: {grid_dim}")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    b, feat_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(grid, coords, align_corners=align_corners, mode="bilinear", padding_mode="border")
    interp = interp.view(b, feat_dim, n).transpose(-1, -2).squeeze()
    return interp


def init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.1,
    b: float = 0.5,
):
    assert in_dim == len(reso)
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for coo_comb in coo_combs:
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    grid_dimensions: int,
    concat_features: bool,
    num_levels: Optional[int],
) -> torch.Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.0
    for grid in ms_grids[:num_levels]:
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            interp_space = interp_space * interp_out_plane
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space
    if concat_features:
        return torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class HexPlaneField(nn.Module):
    def __init__(self, bounds, planeconfig, multires) -> None:
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds], [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        if timestamps is None:
            timestamps = torch.zeros((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
        pts = torch.cat((pts, timestamps), dim=-1).reshape(-1, 4)
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features,
            num_levels=None,
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=pts.device)
        return features


class SegmentTranslationTrajectory(nn.Module):
    def __init__(self, local_times):
        super().__init__()
        if not torch.is_tensor(local_times):
            local_times = torch.tensor(local_times, dtype=torch.float32)
        local_times = local_times.reshape(-1).float()
        if local_times.numel() == 0:
            local_times = torch.zeros(1, dtype=torch.float32)
        self.num_frames = int(local_times.numel())
        self.register_buffer("sample_times", local_times)
        self.translations = nn.Parameter(torch.zeros(self.num_frames, 2))

    def forward(self, local_t):
        if not torch.is_tensor(local_t):
            local_t = torch.tensor(local_t, dtype=self.sample_times.dtype, device=self.sample_times.device)
        local_t = torch.clamp(local_t, 0.0, 1.0).reshape(-1)
        if self.num_frames == 1:
            return self.translations[:1].expand(local_t.shape[0], -1)
        idx = torch.bucketize(local_t, self.sample_times)
        idx = torch.clamp(idx, min=1, max=self.num_frames - 1)
        left_idx = idx - 1
        right_idx = idx
        left_t = self.sample_times[left_idx]
        right_t = self.sample_times[right_idx]
        denom = torch.clamp(right_t - left_t, min=1e-8)
        w = ((local_t - left_t) / denom).unsqueeze(-1)
        left_vals = self.translations[left_idx]
        right_vals = self.translations[right_idx]
        return left_vals + w * (right_vals - left_vals)


class MultiSegmentTranslationTrajectory(nn.Module):
    def __init__(self, num_control_points=4):
        super().__init__()
        self.trajectories = nn.ModuleDict()
        self.segment_specs = {}
        self.frame_to_segment_id = []

    def set_segments(self, segment_specs, frame_to_segment_id):
        self.segment_specs = {int(spec["segment_id"]): spec for spec in segment_specs}
        self.frame_to_segment_id = [int(v) for v in frame_to_segment_id]
        self.trajectories = nn.ModuleDict(
            {
                str(int(spec["segment_id"])): SegmentTranslationTrajectory(spec["local_times"])
                for spec in segment_specs
            }
        )

    def forward(self, segment_id, local_t):
        key = str(int(segment_id))
        if key not in self.trajectories:
            raise KeyError(f"Unknown motion segment {segment_id}")
        return self.trajectories[key](local_t)

    def get_frame_translation(self, frame_idx):
        if not self.frame_to_segment_id:
            raise RuntimeError("Motion segments are not initialized.")
        frame_idx = int(frame_idx)
        segment_id = self.frame_to_segment_id[frame_idx]
        spec = self.segment_specs[int(segment_id)]
        local_idx = frame_idx - int(spec["start_idx"])
        local_t = spec["local_times"][local_idx]
        return self.forward(segment_id, local_t)

    def get_segment_translations(self, segment_id):
        spec = self.segment_specs[int(segment_id)]
        traj = self.trajectories[str(int(segment_id))]
        local_times = torch.tensor(spec["local_times"], dtype=traj.sample_times.dtype, device=traj.sample_times.device)
        return self.forward(segment_id, local_times)

    @torch.no_grad()
    def set_segment_control_points(self, segment_id, control_points):
        key = str(int(segment_id))
        traj = self.trajectories[key]
        cp = torch.as_tensor(control_points, dtype=traj.translations.dtype, device=traj.translations.device)
        if cp.shape != traj.translations.shape:
            raise ValueError(...)
        traj.translations.copy_(cp)


    def has_segments(self):
        return len(self.segment_specs) > 0


class GlobalTranslationTrajectory(nn.Module):
    def __init__(self, num_control_points=5):
        super().__init__()
        self.num_control_points = int(max(2, num_control_points))
        self.register_buffer("knot_times", torch.linspace(0.0, 1.0, self.num_control_points))
        self.control_point_offsets = nn.Parameter(torch.zeros(self.num_control_points - 1, 3))

    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.knot_times.dtype, device=self.knot_times.device)
        t = torch.clamp(t, 0.0, 1.0).reshape(-1)
        control_points = torch.cat(
            [
                torch.zeros((1, 3), dtype=self.control_point_offsets.dtype, device=self.control_point_offsets.device),
                self.control_point_offsets,
            ],
            dim=0,
        )
        idx = torch.bucketize(t, self.knot_times)
        idx = torch.clamp(idx, min=1, max=self.num_control_points - 1)
        left_idx = idx - 1
        right_idx = idx
        left_t = self.knot_times[left_idx]
        right_t = self.knot_times[right_idx]
        denom = torch.clamp(right_t - left_t, min=1e-8)
        w = ((t - left_t) / denom).unsqueeze(-1)
        left_cp = control_points[left_idx]
        right_cp = control_points[right_idx]
        return left_cp + w * (right_cp - left_cp)


class OpacityVelocityField(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=128, vmax=2.0):
        super().__init__()
        self.vmax = float(vmax)
        self.mlp = nn.Sequential(
            nn.Linear(3 + 1 + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, t, z):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t_expand = t.expand(x.shape[0], 1)
        else:
            t_expand = t.reshape(-1, 1)
            if t_expand.shape[0] == 1:
                t_expand = t_expand.expand(x.shape[0], 1)
        z_expand = z.reshape(1, -1).expand(x.shape[0], -1)
        inp = torch.cat([x, t_expand, z_expand], dim=-1)
        v = self.mlp(inp)
        return torch.tanh(v) * self.vmax


class ODEPriorEncoder(nn.Module):
    def __init__(self, point_feat_dim=8, hidden_dim=64, num_frequencies=4):
        super().__init__()
        self.point_feat_dim = int(point_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_frequencies = int(max(1, num_frequencies))
        freqs = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        self.register_buffer("fourier_frequencies", freqs, persistent=False)
        in_dim = 3 + 3 * 2 * self.num_frequencies + self.point_feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _encode_position(self, canonical_pos):
        pieces = [canonical_pos]
        for freq in self.fourier_frequencies.to(dtype=canonical_pos.dtype, device=canonical_pos.device):
            angle = canonical_pos * freq * torch.pi
            pieces.append(torch.sin(angle))
            pieces.append(torch.cos(angle))
        return torch.cat(pieces, dim=-1)

    def forward(self, canonical_pos, point_feat):
        pos_feat = self._encode_position(canonical_pos)
        return self.net(torch.cat([pos_feat, point_feat], dim=-1))


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


def compute_plane_smoothness(t):
    _, _, h, _ = t.shape
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]
    return torch.square(torch.abs(second_difference)).mean()


class Deformation(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        latent_dim=128,
        vmax=0.1,
        opacity_vmax=2.0,
        ode_steps=8,
        trajectory_control_points=4,
        anchor_depth_mode="fixed",
        hex_affects_xyz=False,
        learned_depth_delta=0.0,
    ):
        super().__init__()
        self.kplanes_config = {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 32,
            "resolution": [64, 64, 64, 25],
        }
        self.multires = [1, 2, 4, 8]
        self.grid = HexPlaneField(1.6, self.kplanes_config, self.multires)
        self.feature_out = nn.Sequential(
            nn.Linear(self.grid.feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_deform = nn.Linear(hidden_dim, 3)
        self.scale_deform = nn.Linear(hidden_dim, 3)
        self.rot_deform = nn.Linear(hidden_dim, 4)
        self.opacity_deform = nn.Linear(hidden_dim, 1)

        self.velocity_field = VelocityField(latent_dim=latent_dim, prior_dim=hidden_dim, point_feat_dim=8, vmax=vmax)
        self.ode_prior_encoder = ODEPriorEncoder(point_feat_dim=8, hidden_dim=hidden_dim)
        self.opacity_velocity_field = OpacityVelocityField(latent_dim=latent_dim, vmax=opacity_vmax)
        self.segment_translation_trajectory = MultiSegmentTranslationTrajectory(num_control_points=trajectory_control_points)
        self.ode_steps = int(ode_steps)
        self.ode_grad_frame_window = 2
        self.video_static_latent = None
        self.video_motion_latents = None
        self.canonical_time = 0.0
        self.motion_segments = []
        self.frame_to_segment_id = []
        self.anchor_depth_mode = str(anchor_depth_mode)
        self.hex_affects_xyz = bool(hex_affects_xyz)
        self.anchor_log_depth_scale = nn.Parameter(torch.tensor(float(learned_depth_delta), dtype=torch.float32))

    def set_video_latent(self, z):
        if isinstance(z, dict):
            self.video_static_latent = z.get("static", None)
            self.video_motion_latents = z.get("motion", None)
        else:
            self.video_static_latent = z
            self.video_motion_latents = None

    def set_ode_steps(self, steps):
        self.ode_steps = int(steps)

    def set_ode_grad_frame_window(self, frame_window):
        self.ode_grad_frame_window = int(frame_window)

    def set_vmax(self, vmax):
        self.velocity_field.vmax = float(vmax)

    def set_opacity_vmax(self, vmax):
        self.opacity_velocity_field.vmax = float(vmax)

    def set_canonical_time(self, canonical_time):
        self.canonical_time = float(canonical_time)

    def set_anchor_depth_mode(self, mode):
        self.anchor_depth_mode = str(mode)

    def set_hex_affects_xyz(self, enabled):
        self.hex_affects_xyz = bool(enabled)

    def set_motion_segments(self, motion_segments, frame_to_segment_id):
        self.motion_segments = list(motion_segments)
        self.frame_to_segment_id = [int(v) for v in frame_to_segment_id]
        self.segment_translation_trajectory.set_segments(self.motion_segments, self.frame_to_segment_id)

    def get_global_translation(self, t):
        # Fallback only; the active path uses segment trajectories indexed by frame_idx.
        param = next(self.velocity_field.parameters())
        return torch.zeros((1, 3), dtype=param.dtype, device=param.device)

    def get_relative_global_translation(self, t):
        return self.get_global_translation(t)

    def get_segment_translation(self, frame_idx):
        if not self.segment_translation_trajectory.has_segments():
            param = next(self.velocity_field.parameters())
            return torch.zeros((1, 2), dtype=param.dtype, device=param.device)
        return self.segment_translation_trajectory.get_frame_translation(frame_idx)

    def get_segment_translations(self, segment_id):
        return self.segment_translation_trajectory.get_segment_translations(segment_id)

    def _has_video_context(self):
        return self.video_static_latent is not None

    def _canonical_center(self, points, camera_info):
        if camera_info is not None and camera_info.get("canonical_center", None) is not None:
            return camera_info["canonical_center"].to(dtype=points.dtype, device=points.device).reshape(1, 3)
        return points[:, :3].mean(dim=0, keepdim=True)

    def _time_to_frame_position(self, t):
        if len(self.frame_to_segment_id) > 1:
            return torch.clamp(t, 0.0, 1.0) * float(len(self.frame_to_segment_id) - 1)
        if self.video_motion_latents is not None and self.video_motion_latents.shape[0] > 1:
            return torch.clamp(t, 0.0, 1.0) * float(self.video_motion_latents.shape[0] - 1)
        return torch.zeros_like(t)

    def _num_motion_frames(self):
        if len(self.frame_to_segment_id) > 0:
            return len(self.frame_to_segment_id)
        if self.video_motion_latents is not None and self.video_motion_latents.ndim > 1:
            return int(self.video_motion_latents.shape[0])
        return 1

    def _root_frame_index(self):
        num_frames = self._num_motion_frames()
        if num_frames <= 1:
            return 0
        root = round(self.canonical_time * float(num_frames - 1))
        return int(max(0, min(num_frames - 1, root)))

    def _resolve_target_frame_idx(self, t, frame_idx):
        num_frames = self._num_motion_frames()
        if frame_idx is not None:
            return int(max(0, min(num_frames - 1, int(frame_idx))))
        if num_frames <= 1:
            return 0
        if torch.is_tensor(t):
            t_val = float(t.detach().reshape(-1)[0].item())
        else:
            t_val = float(t)
        t_val = max(0.0, min(1.0, t_val))
        return int(round(t_val * float(num_frames - 1)))

    def _frame_time(self, frame_idx, dtype, device):
        num_frames = self._num_motion_frames()
        if num_frames <= 1:
            return torch.tensor(0.0, dtype=dtype, device=device)
        frame_idx = int(max(0, min(num_frames - 1, int(frame_idx))))
        return torch.tensor(float(frame_idx) / float(num_frames - 1), dtype=dtype, device=device)

    def _translation_at_frame(self, frame_idx, dtype, device):
        if not self.segment_translation_trajectory.has_segments() or len(self.frame_to_segment_id) == 0:
            return torch.zeros((1, 2), dtype=dtype, device=device)
        return self.get_segment_translation(frame_idx).to(dtype=dtype, device=device)

    def _translation_at_time(self, t, dtype, device):
        if not self.segment_translation_trajectory.has_segments() or len(self.frame_to_segment_id) == 0:
            return torch.zeros((1, 2), dtype=dtype, device=device)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = torch.clamp(t, 0.0, 1.0).reshape(1)
        frame_pos = self._time_to_frame_position(t).reshape(1)
        left_idx = int(torch.floor(frame_pos).item())
        right_idx = min(left_idx + 1, len(self.frame_to_segment_id) - 1)
        w = float((frame_pos - left_idx).item())
        left = self.get_segment_translation(left_idx).to(dtype=dtype, device=device)
        right = self.get_segment_translation(right_idx).to(dtype=dtype, device=device)
        return left + w * (right - left)

    def _motion_latent_at_time(self, t, dtype, device):
        if self.video_motion_latents is None:
            return self.video_static_latent.to(dtype=dtype, device=device) if self.video_static_latent is not None else None
        latents = self.video_motion_latents.to(dtype=dtype, device=device)
        if latents.ndim == 1 or latents.shape[0] == 1:
            return latents.reshape(-1)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = torch.clamp(t, 0.0, 1.0).reshape(1)
        frame_pos = self._time_to_frame_position(t).reshape(1)
        left_idx = int(torch.floor(frame_pos).item())
        right_idx = min(left_idx + 1, latents.shape[0] - 1)
        w = float((frame_pos - left_idx).item())
        return latents[left_idx] * (1.0 - w) + latents[right_idx] * w

    def _motion_latent_between_frames(self, frame_idx, next_frame_idx, alpha, dtype, device):
        if self.video_motion_latents is None:
            return self.video_static_latent.to(dtype=dtype, device=device) if self.video_static_latent is not None else None
        latents = self.video_motion_latents.to(dtype=dtype, device=device)
        if latents.ndim == 1 or latents.shape[0] == 1:
            return latents.reshape(-1)
        left_idx = int(max(0, min(latents.shape[0] - 1, int(frame_idx))))
        right_idx = int(max(0, min(latents.shape[0] - 1, int(next_frame_idx))))
        alpha = float(max(0.0, min(1.0, alpha)))
        return latents[left_idx] * (1.0 - alpha) + latents[right_idx] * alpha

    def _hex_prior_features(self, canonical_points, t):
        time_feat = t.expand(canonical_points.shape[0], 1)
        grid_feat = self.grid(canonical_points, time_feat)
        return self.feature_out(grid_feat).float()

    def _ode_prior_features(self, canonical_pos, point_feat):
        return self.ode_prior_encoder(canonical_pos, point_feat).float()

    def _build_control(self, t_cur, shift_2d_cur, shift_2d_next, dt, dtype, device):
        dt_safe = torch.where(dt.abs() < 1e-6, torch.full_like(dt, 1e-6), dt)
        shift_vel = (shift_2d_next - shift_2d_cur) / dt_safe
        t_feat = torch.tensor([[float(t_cur.item())]], dtype=dtype, device=device)
        return torch.cat([shift_2d_cur, shift_vel, t_feat], dim=-1)

    def _zero_mean_acceleration(self, acc, weights):
        weights = weights / torch.clamp(weights.sum(), min=1e-6)
        mean_acc = (acc * weights).sum(dim=0, keepdim=True)
        return acc - mean_acc

    def _integrate_frame_transition(
        self,
        canonical_points,
        canonical_center,
        canonical_pos,
        point_feat,
        ode_prior_feat,
        weights,
        residual_pos,
        residual_vel,
        frame_idx,
        next_frame_idx,
        camera_info,
        return_velocities=False,
    ):
        dtype = canonical_points.dtype
        device = canonical_points.device
        t_cur = self._frame_time(frame_idx, dtype, device)
        t_next = self._frame_time(next_frame_idx, dtype, device)
        shift_2d_cur = self._translation_at_frame(frame_idx, dtype, device)
        shift_2d_next = self._translation_at_frame(next_frame_idx, dtype, device)
        anchor_cur = canonical_center + self._project_image_shift_to_world(shift_2d_cur, canonical_points, camera_info)
        anchor_next = canonical_center + self._project_image_shift_to_world(shift_2d_next, canonical_points, camera_info)
        dt_frame = t_next - t_cur
        dt_frame_safe = torch.where(dt_frame.abs() < 1e-6, torch.full_like(dt_frame, 1e-6), dt_frame)
        anchor_vel = (anchor_next - anchor_cur) / dt_frame_safe
        velocity_stats = None
        velocity_steps = 0
        if return_velocities:
            velocity_stats = {
                "xyz": torch.zeros((), dtype=dtype, device=device),
                "residual_xyz": torch.zeros((), dtype=dtype, device=device),
                "residual_acc": torch.zeros((), dtype=dtype, device=device),
            }

        if float(torch.abs(dt_frame).item()) < 1e-8:
            return residual_pos, residual_vel, anchor_vel, (velocity_stats or {})

        z_static = self.video_static_latent.to(dtype=dtype, device=device)
        num_substeps = max(int(self.ode_steps), 1)
        dt = dt_frame / float(num_substeps)
        for substep in range(num_substeps):
            alpha_cur = float(substep) / float(num_substeps)
            alpha_next = float(substep + 1) / float(num_substeps)
            t_step = t_cur + dt_frame * alpha_cur
            shift_step = shift_2d_cur * (1.0 - alpha_cur) + shift_2d_next * alpha_cur
            shift_step_next = shift_2d_cur * (1.0 - alpha_next) + shift_2d_next * alpha_next
            control = self._build_control(t_step, shift_step, shift_step_next, dt, dtype, device)
            z_motion = self._motion_latent_between_frames(frame_idx, next_frame_idx, alpha_cur, dtype, device)
            acc = self.velocity_field(
                residual_pos,
                residual_vel,
                canonical_pos,
                point_feat,
                ode_prior_feat,
                control,
                z_static,
                z_motion,
            )
            acc = self._zero_mean_acceleration(acc, weights)
            residual_vel = residual_vel + dt * acc
            residual_pos = residual_pos + dt * residual_vel
            if return_velocities:
                total_velocity = anchor_vel.expand_as(residual_vel) + residual_vel
                velocity_stats["xyz"] = velocity_stats["xyz"] + (total_velocity ** 2).mean()
                velocity_stats["residual_xyz"] = velocity_stats["residual_xyz"] + (residual_vel ** 2).mean()
                velocity_stats["residual_acc"] = velocity_stats["residual_acc"] + (acc ** 2).mean()
                velocity_steps += 1

        if return_velocities and velocity_steps > 0:
            inv_steps = 1.0 / float(velocity_steps)
            velocity_stats = {key: value * inv_steps for key, value in velocity_stats.items()}
        elif not return_velocities:
            velocity_stats = {}

        return residual_pos, residual_vel, anchor_vel, velocity_stats

    def _rollout_residual_sonode(self, points, scales, rotations, opacity, target_frame_idx, camera_info, return_velocities=False):
        if not self._has_video_context():
            zero_vel = torch.zeros_like(points[:, :3])
            return points[:, :3], [], zero_vel, zero_vel

        canonical_points = points[:, :3]
        canonical_center = self._canonical_center(canonical_points, camera_info)
        canonical_pos = canonical_points - canonical_center
        point_feat = torch.cat([scales[:, :3], rotations[:, :4], opacity[:, :1]], dim=-1)
        ode_prior_feat = self._ode_prior_features(canonical_pos, point_feat)
        weights = torch.sigmoid(opacity[:, :1]).detach()
        residual_pos = torch.zeros_like(canonical_points)
        residual_vel = torch.zeros_like(canonical_points)
        velocity_stats = None
        velocity_frame_steps = 0
        final_anchor_vel = torch.zeros_like(canonical_points)
        root_frame_idx = self._root_frame_index()
        current_frame_idx = root_frame_idx
        step_direction = 1 if target_frame_idx >= root_frame_idx else -1
        total_frame_steps = abs(int(target_frame_idx) - int(root_frame_idx))
        grad_window = int(self.ode_grad_frame_window)
        warmup_steps = 0
        if torch.is_grad_enabled() and grad_window > 0 and total_frame_steps > grad_window:
            warmup_steps = total_frame_steps - grad_window

        for _ in range(warmup_steps):
            next_frame_idx = current_frame_idx + step_direction
            with torch.no_grad():
                residual_pos, residual_vel, final_anchor_vel, step_velocities = self._integrate_frame_transition(
                    canonical_points,
                    canonical_center,
                    canonical_pos,
                    point_feat,
                    ode_prior_feat,
                    weights,
                    residual_pos,
                    residual_vel,
                    current_frame_idx,
                    next_frame_idx,
                    camera_info,
                    return_velocities=return_velocities,
                )
            residual_pos = residual_pos.detach()
            residual_vel = residual_vel.detach()
            if return_velocities and len(step_velocities) > 0:
                if velocity_stats is None:
                    velocity_stats = {
                        key: torch.zeros_like(value) for key, value in step_velocities.items()
                    }
                for key, value in step_velocities.items():
                    velocity_stats[key] = velocity_stats[key] + value
                velocity_frame_steps += 1
            current_frame_idx = next_frame_idx

        while current_frame_idx != target_frame_idx:
            next_frame_idx = current_frame_idx + step_direction
            residual_pos, residual_vel, final_anchor_vel, step_velocities = self._integrate_frame_transition(
                canonical_points,
                canonical_center,
                canonical_pos,
                point_feat,
                ode_prior_feat,
                weights,
                residual_pos,
                residual_vel,
                current_frame_idx,
                next_frame_idx,
                camera_info,
                return_velocities=return_velocities,
            )
            if return_velocities and len(step_velocities) > 0:
                if velocity_stats is None:
                    velocity_stats = {
                        key: torch.zeros_like(value) for key, value in step_velocities.items()
                    }
                for key, value in step_velocities.items():
                    velocity_stats[key] = velocity_stats[key] + value
                velocity_frame_steps += 1
            current_frame_idx = next_frame_idx

        shift_end = self._translation_at_frame(target_frame_idx, points.dtype, points.device)
        anchor_world = canonical_center + self._project_image_shift_to_world(shift_end, points, camera_info)
        world_xyz = anchor_world.expand_as(canonical_points) + canonical_pos + residual_pos
        if return_velocities and velocity_stats is not None and velocity_frame_steps > 0:
            inv_steps = 1.0 / float(velocity_frame_steps)
            velocity_stats = {key: value * inv_steps for key, value in velocity_stats.items()}
        elif not return_velocities:
            velocity_stats = []
        else:
            zero_stat = torch.zeros((), dtype=points.dtype, device=points.device)
            velocity_stats = {
                "xyz": zero_stat,
                "residual_xyz": zero_stat,
                "residual_acc": zero_stat,
            }
        return world_xyz, velocity_stats, residual_vel, final_anchor_vel

    def forward_dynamic(self, points, scales, rotations, opacity, t, vis_ratio, return_velocities=False, motion_mode="joint", frame_idx=None, camera_info=None):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=points.dtype, device=points.device)
        t = torch.clamp(t, 0.0, 1.0)
        use_hex = motion_mode in ("hex", "joint")
        use_ode = motion_mode in ("ode", "joint")
        target_frame_idx = self._resolve_target_frame_idx(t, frame_idx)
        shift_2d = self._translation_at_frame(target_frame_idx, points.dtype, points.device)
        canonical_center = self._canonical_center(points[:, :3], camera_info)
        anchor_world = canonical_center + self._project_image_shift_to_world(shift_2d, points, camera_info)
        base_xyz = anchor_world.expand_as(points[:, :3]) + (points[:, :3] - canonical_center)

        time_feat = t.expand(points.shape[0], 1)
        prior_hidden = self._hex_prior_features(points[:, :3], time_feat)
        if use_hex:
            if self.hex_affects_xyz:
                dx_hex = self.pos_deform(prior_hidden)
            else:
                dx_hex = torch.zeros_like(points[:, :3])
            ds = self.scale_deform(prior_hidden)
            dr = self.rot_deform(prior_hidden)
            do = self.opacity_deform(prior_hidden)
        else:
            dx_hex = torch.zeros_like(points[:, :3])
            ds = torch.zeros_like(scales[:, :3])
            dr = torch.zeros_like(rotations[:, :4])
            do = torch.zeros_like(opacity[:, :1])

        if use_ode and self._has_video_context():
            pts_ode, velocities, _, _ = self._rollout_residual_sonode(
                points,
                scales,
                rotations,
                opacity,
                target_frame_idx,
                camera_info,
                return_velocities=return_velocities,
            )
            pts = pts_ode + dx_hex
        else:
            pts = base_xyz + dx_hex
            velocities = []

        scales_out = scales[:, :3] + ds
        rotations_out = rotations[:, :4] + dr
        opacity_out = opacity[:, :1] + do
        if return_velocities:
            return pts, scales_out, rotations_out, opacity_out, velocities
        return pts, scales_out, rotations_out, opacity_out, []

    def _project_image_shift_to_world(self, shift_2d, points, camera_info):
        if camera_info is None:
            zeros_z = torch.zeros((shift_2d.shape[0], 1), dtype=shift_2d.dtype, device=shift_2d.device)
            return torch.cat([shift_2d, zeros_z], dim=-1)
        right = camera_info["right"].to(dtype=points.dtype, device=points.device).reshape(1, 3)
        up = camera_info["up"].to(dtype=points.dtype, device=points.device).reshape(1, 3)
        depth_scale = self._resolve_anchor_depth_scale(camera_info, points).reshape(1, 1)
        x_scale = 2.0 * depth_scale * float(camera_info["tan_half_fovx"])
        y_scale = 2.0 * depth_scale * float(camera_info["tan_half_fovy"])
        dx_world = shift_2d[:, :1] * x_scale * right
        dy_world = -shift_2d[:, 1:2] * y_scale * up
        return dx_world + dy_world

    def _resolve_anchor_depth_scale(self, camera_info, points):
        fallback = torch.as_tensor(camera_info["depth_scale"], dtype=points.dtype, device=points.device)
        mode = self.anchor_depth_mode.lower().strip()
        if mode == "gt":
            anchor_depth = camera_info.get("anchor_depth", None)
            if anchor_depth is not None:
                return torch.as_tensor(anchor_depth, dtype=points.dtype, device=points.device)
            return fallback
        if mode == "learned_delta":
            return fallback * torch.exp(self.anchor_log_depth_scale.to(dtype=points.dtype, device=points.device))
        return fallback

    def query_velocity(self, points, t, frame_idx=None, camera_info=None):
        if not self._has_video_context():
            return torch.zeros_like(points[:, :3])
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=points.dtype, device=points.device)
        t = torch.clamp(t, 0.0, 1.0)
        target_frame_idx = self._resolve_target_frame_idx(t, frame_idx)
        dummy_scales = torch.zeros((points.shape[0], 3), dtype=points.dtype, device=points.device)
        dummy_rot = torch.zeros((points.shape[0], 4), dtype=points.dtype, device=points.device)
        dummy_opacity = torch.zeros((points.shape[0], 1), dtype=points.dtype, device=points.device)
        _, _, residual_vel, anchor_vel = self._rollout_residual_sonode(
            points,
            dummy_scales,
            dummy_rot,
            dummy_opacity,
            target_frame_idx,
            camera_info,
            return_velocities=False,
        )
        return anchor_vel.expand_as(residual_vel) + residual_vel

    def project_image_shift_to_world(self, shift_2d, points, camera_info):
        return self._project_image_shift_to_world(shift_2d, points, camera_info)


class deform_network(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        vmax=0.1,
        opacity_vmax=2.0,
        ode_steps=8,
        trajectory_control_points=4,
        anchor_depth_mode="fixed",
        hex_affects_xyz=False,
        learned_depth_delta=0.0,
    ):
        super().__init__()
        self.video_encoder = VideoEncoder(latent_dim=latent_dim)
        self.deformation_net = Deformation(
            latent_dim=latent_dim,
            vmax=vmax,
            opacity_vmax=opacity_vmax,
            ode_steps=ode_steps,
            trajectory_control_points=trajectory_control_points,
            anchor_depth_mode=anchor_depth_mode,
            hex_affects_xyz=hex_affects_xyz,
            learned_depth_delta=learned_depth_delta,
        )

    def set_video_frames(self, frames):
        z = self.video_encoder(frames)
        self.deformation_net.set_video_latent(z)
        return z

    def set_video_latent(self, z):
        self.deformation_net.set_video_latent(z)

    def set_ode_steps(self, steps):
        self.deformation_net.set_ode_steps(steps)

    def set_ode_grad_frame_window(self, frame_window):
        self.deformation_net.set_ode_grad_frame_window(frame_window)

    def set_vmax(self, vmax):
        self.deformation_net.set_vmax(vmax)

    def set_opacity_vmax(self, vmax):
        self.deformation_net.set_opacity_vmax(vmax)

    def set_canonical_time(self, canonical_time):
        self.deformation_net.set_canonical_time(canonical_time)

    def set_anchor_depth_mode(self, mode):
        self.deformation_net.set_anchor_depth_mode(mode)

    def set_hex_affects_xyz(self, enabled):
        self.deformation_net.set_hex_affects_xyz(enabled)

    def set_motion_segments(self, motion_segments, frame_to_segment_id):
        self.deformation_net.set_motion_segments(motion_segments, frame_to_segment_id)

    def get_global_translation(self, t):
        return self.deformation_net.get_global_translation(t)

    def get_relative_global_translation(self, t):
        return self.deformation_net.get_relative_global_translation(t)

    def get_segment_translation(self, frame_idx):
        return self.deformation_net.get_segment_translation(frame_idx)

    def get_segment_translations(self, segment_id):
        return self.deformation_net.get_segment_translations(segment_id)

    def set_segment_control_points(self, segment_id, control_points):
        self.deformation_net.segment_translation_trajectory.set_segment_control_points(segment_id, control_points)

    def query_velocity(self, points, times_sel, frame_idx=None, camera_info=None):
        return self.deformation_net.query_velocity(points, times_sel, frame_idx=frame_idx, camera_info=camera_info)

    def project_image_shift_to_world(self, shift_2d, points, camera_info):
        return self.deformation_net.project_image_shift_to_world(shift_2d, points, camera_info)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None, vis_ratio=1.0, return_velocities=False, motion_mode="joint", frame_idx=None, camera_info=None):
        if times_sel is None:
            return point[:, :3], scales, rotations, opacity, []
        return self.deformation_net.forward_dynamic(
            point,
            scales=scales,
            rotations=rotations,
            opacity=opacity,
            t=times_sel,
            vis_ratio=vis_ratio,
            return_velocities=return_velocities,
            motion_mode=motion_mode,
            frame_idx=frame_idx,
            camera_info=camera_info,
        )

    def deform_positions(self, point, times_sel, vis_ratio=1.0, return_velocities=False, motion_mode="joint", frame_idx=None, camera_info=None):
        dummy_scales = torch.zeros((point.shape[0], 3), device=point.device, dtype=point.dtype)
        dummy_rot = torch.zeros((point.shape[0], 4), device=point.device, dtype=point.dtype)
        dummy_op = torch.zeros((point.shape[0], 1), device=point.device, dtype=point.dtype)
        x, _, _, _, velocities = self.forward(
            point,
            scales=dummy_scales,
            rotations=dummy_rot,
            opacity=dummy_op,
            times_sel=times_sel,
            vis_ratio=vis_ratio,
            return_velocities=return_velocities,
            motion_mode=motion_mode,
            frame_idx=frame_idx,
            camera_info=camera_info,
        )
        return x, velocities

    def get_hex_mlp_parameters(self):
        params = list(self.deformation_net.feature_out.parameters())
        if self.deformation_net.hex_affects_xyz:
            params += list(self.deformation_net.pos_deform.parameters())
        params += list(self.deformation_net.scale_deform.parameters())
        params += list(self.deformation_net.rot_deform.parameters())
        params += list(self.deformation_net.opacity_deform.parameters())
        return params

    def get_hex_grid_parameters(self):
        return list(self.deformation_net.grid.parameters())

    def get_trajectory_parameters(self):
        return list(self.deformation_net.segment_translation_trajectory.parameters())

    def get_ode_parameters(self):
        return (
            list(self.deformation_net.velocity_field.parameters())
            + list(self.deformation_net.ode_prior_encoder.parameters())
            + [self.deformation_net.anchor_log_depth_scale]
        )

    def get_encoder_parameters(self):
        return list(self.video_encoder.parameters())

    def get_mlp_parameters(self):
        return self.get_hex_mlp_parameters() + self.get_ode_parameters()

    def get_grid_parameters(self):
        return self.get_hex_grid_parameters()
