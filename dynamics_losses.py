import torch


def _infer_device_from_velocity(velocities):
    if torch.is_tensor(velocities):
        return velocities.device
    if isinstance(velocities, dict):
        for value in velocities.values():
            if torch.is_tensor(value):
                return value.device
    if isinstance(velocities, (list, tuple)):
        for value in velocities:
            device = _infer_device_from_velocity(value)
            if device is not None:
                return device
    return None


def velocity_reg(velocities):
    device = _infer_device_from_velocity(velocities)
    if velocities is None:
        return torch.tensor(0.0, device=device)
    if isinstance(velocities, (list, tuple)):
        if len(velocities) == 0:
            return torch.tensor(0.0, device=device)
        vals = []
        for v in velocities:
            if isinstance(v, dict):
                vals.extend((value ** 2).mean() for value in v.values())
            else:
                vals.append((v ** 2).mean())
        if len(vals) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(vals).mean()
    if isinstance(velocities, dict):
        vals = [(v ** 2).mean() for v in velocities.values()]
        if len(vals) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(vals).mean()
    return (velocities ** 2).mean()


def curvature_loss(x_prev, x_mid, x_next):
    curv = x_next - 2.0 * x_mid + x_prev
    return (curv ** 2).mean()


def anchor_shape_loss(x_t, x_anchor):
    return torch.mean((x_t - x_anchor) ** 2)


def cycle_sparse_loss(x_exit, x_reentry):
    return torch.mean((x_exit - x_reentry) ** 2)
