import torch
import torch.nn as nn


def _expand_time_like(t, n, dtype, device):
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=dtype, device=device)
    if t.ndim == 0:
        return t.expand(n, 1)
    t = t.reshape(-1, 1).to(dtype=dtype, device=device)
    if t.shape[0] == 1:
        return t.expand(n, 1)
    return t


def _expand_context_like(vec, n, dtype, device):
    if vec is None:
        return None
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec, dtype=dtype, device=device)
    vec = vec.to(dtype=dtype, device=device)
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    if vec.shape[0] == 1:
        vec = vec.expand(n, -1)
    return vec


class VelocityField(nn.Module):
    """Residual SONODE acceleration field.

    The field predicts residual acceleration rather than absolute world velocity.
    Inputs are arranged around the decomposition:
        x(t) = A(t) + p_can + r(t)
    where r(t) is the local residual and v(t) = dr/dt is its velocity.
    """

    def __init__(
        self,
        latent_dim=128,
        prior_dim=64,
        control_dim=5,
        point_feat_dim=8,
        hidden_dim=192,
        vmax=0.1,
    ):
        super().__init__()
        self.vmax = float(vmax)
        in_dim = 3 + 3 + 3 + point_feat_dim + prior_dim + control_dim + latent_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, residual_pos, residual_vel, canonical_pos, point_feat, prior_feat, control, z_static, z_motion):
        n = residual_pos.shape[0]
        dtype = residual_pos.dtype
        device = residual_pos.device

        point_feat = _expand_context_like(point_feat, n, dtype, device)
        prior_feat = _expand_context_like(prior_feat, n, dtype, device)
        control = _expand_context_like(control, n, dtype, device)
        z_static = _expand_context_like(z_static, n, dtype, device)
        z_motion = _expand_context_like(z_motion, n, dtype, device)

        pieces = [residual_pos, residual_vel, canonical_pos]
        if point_feat is not None:
            pieces.append(point_feat)
        if prior_feat is not None:
            pieces.append(prior_feat)
        if control is not None:
            pieces.append(control)
        if z_static is not None:
            pieces.append(z_static)
        if z_motion is not None:
            pieces.append(z_motion)

        acc = self.net(torch.cat(pieces, dim=-1))
        return torch.tanh(acc) * self.vmax
