from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import Tensor


def raster_to_drivable_mask(rasterized_da: np.ndarray) -> np.ndarray:
    """Convert a stored raster layer to a binary drivable mask."""
    raster = np.asarray(rasterized_da)
    if raster.dtype == np.bool_:
        return raster.copy()
    return raster > 0


def build_signed_distance_field(
    drivable_mask: np.ndarray,
    meters_per_pixel: float = 1.0,
) -> np.ndarray:
    """Build an SDF in meters: negative on-road, positive off-road."""
    if meters_per_pixel <= 0:
        raise ValueError("meters_per_pixel must be positive.")

    drivable_mask = np.asarray(drivable_mask, dtype=bool)
    inside_distance = distance_transform_edt(drivable_mask, sampling=meters_per_pixel)
    outside_distance = distance_transform_edt(~drivable_mask, sampling=meters_per_pixel)
    return (outside_distance - inside_distance).astype(np.float32)


def build_signed_distance_field_from_raster(
    rasterized_da: np.ndarray,
    sim2_s: np.ndarray | float,
) -> np.ndarray:
    """Build an SDF in meters from the stored raster and Sim2 scale."""
    scale = float(np.asarray(sim2_s).reshape(-1)[0])
    if scale <= 0:
        raise ValueError("sim2_s must be positive.")
    meters_per_pixel = 1.0 / scale
    drivable_mask = raster_to_drivable_mask(rasterized_da)
    return build_signed_distance_field(drivable_mask, meters_per_pixel=meters_per_pixel)


def _ensure_batched_2d(points_world: Tensor) -> Tuple[Tensor, bool]:
    if points_world.ndim == 2:
        return points_world.unsqueeze(0), True
    return points_world, False


def _ensure_batched_map(sdf_map: Tensor) -> Tuple[Tensor, bool]:
    if sdf_map.ndim == 2:
        return sdf_map.unsqueeze(0), True
    return sdf_map, False


def _ensure_batch_tensor(x: Tensor, batch_size: int) -> Tensor:
    if x.ndim == 0:
        return x.view(1).expand(batch_size)
    if x.ndim >= 1 and x.shape[0] == batch_size:
        return x
    return x.unsqueeze(0).expand(batch_size, *x.shape)


def _ensure_orig_dims(orig_dims: Optional[Tensor], batch_size: int, device: torch.device) -> Optional[Tensor]:
    if orig_dims is None:
        return None

    orig_dims = orig_dims.to(device=device, dtype=torch.long)
    if orig_dims.ndim == 1:
        orig_dims = orig_dims.unsqueeze(0)

    if orig_dims.shape[0] != batch_size:
        if orig_dims.shape[0] == 1:
            orig_dims = orig_dims.expand(batch_size, -1)
        else:
            raise ValueError("orig_dims batch size must match sdf_map batch size.")

    return orig_dims


def world_to_pixel_xy(
    points_world: Tensor,
    sim2_R: Tensor,
    sim2_t: Tensor,
    sim2_s: Tensor,
) -> Tensor:
    """Transform world xy coordinates to pixel xy using full Sim2."""
    points_world, squeezed = _ensure_batched_2d(points_world.to(torch.float32))
    batch_size = points_world.shape[0]

    sim2_R = _ensure_batch_tensor(sim2_R.to(points_world.device, dtype=torch.float32), batch_size)
    sim2_t = _ensure_batch_tensor(sim2_t.to(points_world.device, dtype=torch.float32), batch_size)
    sim2_s = _ensure_batch_tensor(sim2_s.to(points_world.device, dtype=torch.float32), batch_size).reshape(batch_size, -1)
    sim2_s = sim2_s[:, 0].view(batch_size, 1, 1)

    pixel_xy = sim2_s * torch.matmul(points_world, sim2_R.transpose(-1, -2)) + sim2_t.unsqueeze(1)
    return pixel_xy.squeeze(0) if squeezed else pixel_xy


def _pixel_xy_to_grid(pixel_xy: Tensor, height: int, width: int) -> Tensor:
    if width <= 1 or height <= 1:
        raise ValueError("SDF map height and width must both be greater than 1 for bilinear sampling.")

    x = pixel_xy[..., 0]
    y = pixel_xy[..., 1]
    x_norm = (2.0 * x / (width - 1)) - 1.0
    y_norm = (2.0 * y / (height - 1)) - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def sample_sdf_at_world_points(
    sdf_map: Tensor,
    points_world: Tensor,
    sim2_R: Tensor,
    sim2_t: Tensor,
    sim2_s: Tensor,
    orig_dims: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample an SDF map at world points with bilinear interpolation."""
    sdf_map, squeezed_map = _ensure_batched_map(sdf_map.to(torch.float32))
    points_world, squeezed_points = _ensure_batched_2d(points_world.to(torch.float32))
    if squeezed_map != squeezed_points and sdf_map.shape[0] != points_world.shape[0]:
        raise ValueError("sdf_map and points_world batch dimensions do not align.")

    batch_size, height, width = sdf_map.shape
    if points_world.shape[0] != batch_size:
        if batch_size == 1:
            sdf_map = sdf_map.expand(points_world.shape[0], -1, -1)
            batch_size = points_world.shape[0]
        else:
            raise ValueError("points_world batch size must match sdf_map batch size.")

    pixel_xy = world_to_pixel_xy(points_world, sim2_R, sim2_t, sim2_s)
    x = pixel_xy[..., 0]
    y = pixel_xy[..., 1]

    orig_dims = _ensure_orig_dims(orig_dims, batch_size=batch_size, device=pixel_xy.device)
    if orig_dims is None:
        max_x = torch.full_like(x, width - 1)
        max_y = torch.full_like(y, height - 1)
    else:
        valid_heights = orig_dims[:, 0].clamp(min=1, max=height).to(device=pixel_xy.device, dtype=x.dtype)
        valid_widths = orig_dims[:, 1].clamp(min=1, max=width).to(device=pixel_xy.device, dtype=x.dtype)
        max_x = (valid_widths - 1.0).unsqueeze(-1)
        max_y = (valid_heights - 1.0).unsqueeze(-1)

    in_bounds = (x >= 0.0) & (x <= max_x) & (y >= 0.0) & (y <= max_y)

    x_clamped = torch.minimum(torch.maximum(x, torch.zeros_like(x)), max_x)
    y_clamped = torch.minimum(torch.maximum(y, torch.zeros_like(y)), max_y)
    clamped_xy = torch.stack((x_clamped, y_clamped), dim=-1)

    grid = _pixel_xy_to_grid(clamped_xy, height=height, width=width).unsqueeze(2)
    sampled_sdf = F.grid_sample(
        sdf_map.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(1).squeeze(-1)

    overflow_x_px = (-x).clamp_min(0.0) + (x - max_x).clamp_min(0.0)
    overflow_y_px = (-y).clamp_min(0.0) + (y - max_y).clamp_min(0.0)
    overflow_px = torch.sqrt(overflow_x_px.square() + overflow_y_px.square())

    sim2_s = _ensure_batch_tensor(sim2_s.to(sampled_sdf.device, dtype=sampled_sdf.dtype), batch_size).reshape(batch_size, -1)
    pixels_per_meter = sim2_s[:, 0].clamp_min(1e-6).unsqueeze(-1)
    overflow_m = overflow_px / pixels_per_meter

    offroad_distance = sampled_sdf.clamp_min(0.0) + overflow_m

    if squeezed_map and squeezed_points:
        return sampled_sdf.squeeze(0), offroad_distance.squeeze(0), in_bounds.squeeze(0)
    return sampled_sdf, offroad_distance, in_bounds
