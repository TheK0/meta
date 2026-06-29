from __future__ import annotations

from typing import Literal

import torch


def _require_dim_zero(dim: int) -> None:
    if dim != 0:
        raise NotImplementedError("torch_scatter compatibility only supports dim=0.")


def _normalize_dim_size(index: torch.Tensor, dim_size: int | None) -> int:
    if dim_size is not None:
        return int(dim_size)
    if index.numel() == 0:
        return 0
    return int(index.max().item()) + 1


def _broadcast_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if index.dim() != 1:
        raise NotImplementedError("torch_scatter compatibility expects a 1D index tensor.")
    view_shape = (index.shape[0],) + (1,) * (src.dim() - 1)
    return index.view(view_shape).expand_as(src)


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    _require_dim_zero(dim)
    output_size = _normalize_dim_size(index, dim_size)
    out_shape = (output_size,) + tuple(src.shape[1:])
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    if index.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    _require_dim_zero(dim)
    sums = scatter_sum(src=src, index=index, dim=dim, dim_size=dim_size)
    counts = torch.zeros(
        (sums.shape[0],),
        dtype=torch.int64,
        device=src.device,
    )
    if index.numel() > 0:
        counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.int64))
    safe_counts = counts.clamp_min(1)
    divisor = safe_counts.view((-1,) + (1,) * (src.dim() - 1))
    mean = sums / divisor
    if counts.numel() > 0:
        mean = mean.masked_fill((counts == 0).view((-1,) + (1,) * (src.dim() - 1)), 0)
    return mean


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
):
    _require_dim_zero(dim)
    output_size = _normalize_dim_size(index, dim_size)
    out_shape = (output_size,) + tuple(src.shape[1:])
    values = torch.full(
        out_shape,
        fill_value=float("-inf") if src.dtype.is_floating_point else torch.iinfo(src.dtype).min,
        dtype=src.dtype,
        device=src.device,
    )
    argmax = torch.full(out_shape, fill_value=-1, dtype=torch.long, device=src.device)
    if index.numel() == 0:
        return values, argmax

    broadcast_index = _broadcast_index(index, src)
    values.scatter_reduce_(0, broadcast_index, src, reduce="amax", include_self=True)

    gathered_values = values.index_select(0, index)
    position_dtype = torch.float32 if src.device.type == "mps" else torch.float64
    positions = torch.arange(src.shape[0], dtype=position_dtype, device=src.device)
    expanded_positions = _broadcast_index(positions, src)
    sentinel_value = float(src.shape[0])
    sentinel = torch.full_like(expanded_positions, sentinel_value)
    candidates = torch.where(src == gathered_values, expanded_positions, sentinel)
    argmax_candidates = torch.full(
        out_shape,
        fill_value=sentinel_value,
        dtype=position_dtype,
        device=src.device,
    )
    argmax_candidates.scatter_reduce_(
        0,
        broadcast_index,
        candidates,
        reduce="amin",
        include_self=True,
    )
    argmax = argmax_candidates.to(torch.long)
    argmax.masked_fill_(argmax == src.shape[0], -1)
    return values, argmax


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
    reduce: Literal["sum", "mean", "max", "min"] = "sum",
) -> torch.Tensor:
    _require_dim_zero(dim)
    if reduce == "sum":
        return scatter_sum(src=src, index=index, dim=dim, dim_size=dim_size)
    if reduce == "mean":
        return scatter_mean(src=src, index=index, dim=dim, dim_size=dim_size)
    if reduce == "max":
        return scatter_max(src=src, index=index, dim=dim, dim_size=dim_size)[0]
    if reduce == "min":
        output_size = _normalize_dim_size(index, dim_size)
        out_shape = (output_size,) + tuple(src.shape[1:])
        values = torch.full(
            out_shape,
            fill_value=torch.finfo(src.dtype).max if src.dtype.is_floating_point else torch.iinfo(src.dtype).max,
            dtype=src.dtype,
            device=src.device,
        )
        if index.numel() == 0:
            return values
        values.scatter_reduce_(
            0,
            _broadcast_index(index, src),
            src,
            reduce="amin",
            include_self=True,
        )
        return values
    raise ValueError(f"Unsupported scatter reduction: {reduce}")


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    _require_dim_zero(dim)
    dim_size = _normalize_dim_size(index, None)
    max_per_group = scatter_max(src=src, index=index, dim=dim, dim_size=dim_size)[0]
    centered = src - max_per_group.index_select(0, index)
    exp_values = torch.exp(centered)
    denom = scatter_sum(src=exp_values, index=index, dim=dim, dim_size=dim_size)
    return exp_values / denom.index_select(0, index)


def scatter_log_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    _require_dim_zero(dim)
    dim_size = _normalize_dim_size(index, None)
    max_per_group = scatter_max(src=src, index=index, dim=dim, dim_size=dim_size)[0]
    centered = src - max_per_group.index_select(0, index)
    exp_values = torch.exp(centered)
    denom = scatter_sum(src=exp_values, index=index, dim=dim, dim_size=dim_size)
    return centered - torch.log(denom.index_select(0, index))
