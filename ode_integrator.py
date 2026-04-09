import torch


def integrate_euler(x0, t0, t1, velocity_field, z, steps=8, return_velocities=False):
    if steps <= 0:
        raise ValueError("steps must be > 0")

    if not torch.is_tensor(t0):
        t0 = torch.tensor(t0, dtype=x0.dtype, device=x0.device)
    if not torch.is_tensor(t1):
        t1 = torch.tensor(t1, dtype=x0.dtype, device=x0.device)

    dt = (t1 - t0) / float(steps)
    x = x0
    t = t0
    velocities = []

    for _ in range(steps):
        v = velocity_field(x, t, z)
        if return_velocities:
            velocities.append(v)
        x = x + dt * v
        t = t + dt

    if return_velocities:
        return x, velocities
    return x
