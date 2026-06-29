from __future__ import annotations

import torch

from fsmol_cliff.fsmol_bridge import _apply_source_patches
from fsmol_cliff.torch_scatter_compat import (
    scatter,
    scatter_log_softmax,
    scatter_max,
    scatter_mean,
    scatter_softmax,
    scatter_sum,
)


def test_apply_source_patches_rewrites_gnn_torch_scatter_import() -> None:
    source = "from torch_scatter import scatter_sum, scatter_log_softmax, scatter_mean, scatter_max\n"

    patched = _apply_source_patches("fs_mol.modules.gnn", source)

    assert (
        "from fsmol_cliff.torch_scatter_compat import "
        "scatter_sum, scatter_log_softmax, scatter_mean, scatter_max"
    ) in patched


def test_apply_source_patches_rewrites_graph_readout_torch_scatter_import() -> None:
    source = "from torch_scatter import scatter_softmax, scatter\n"

    patched = _apply_source_patches("fs_mol.modules.graph_readout", source)

    assert "from fsmol_cliff.torch_scatter_compat import scatter_softmax, scatter" in patched


def test_scatter_sum_and_mean_aggregate_rows_by_index() -> None:
    src = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=torch.float32)
    index = torch.tensor([0, 1, 0], dtype=torch.long)

    summed = scatter_sum(src=src, index=index, dim=0, dim_size=3)
    meaned = scatter_mean(src=src, index=index, dim=0, dim_size=3)

    assert torch.equal(
        summed,
        torch.tensor([[8.0, 13.0], [3.0, 5.0], [0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.equal(
        meaned,
        torch.tensor([[4.0, 6.5], [3.0, 5.0], [0.0, 0.0]], dtype=torch.float32),
    )


def test_scatter_max_returns_values_and_indices() -> None:
    src = torch.tensor([[1.0, 2.0], [4.0, 3.0], [0.5, 8.0]], dtype=torch.float32)
    index = torch.tensor([0, 0, 1], dtype=torch.long)

    values, argmax = scatter_max(src=src, index=index, dim=0, dim_size=3)

    assert torch.equal(
        values,
        torch.tensor([[4.0, 3.0], [0.5, 8.0], [float("-inf"), float("-inf")]], dtype=torch.float32),
    )
    assert torch.equal(
        argmax,
        torch.tensor([[1, 1], [2, 2], [-1, -1]], dtype=torch.long),
    )


def test_scatter_dispatches_supported_reductions() -> None:
    src = torch.tensor([[1.0, 4.0], [3.0, 2.0], [6.0, 0.5]], dtype=torch.float32)
    index = torch.tensor([0, 0, 1], dtype=torch.long)

    summed = scatter(src=src, index=index, dim=0, dim_size=2, reduce="sum")
    maxed = scatter(src=src, index=index, dim=0, dim_size=2, reduce="max")
    mined = scatter(src=src, index=index, dim=0, dim_size=2, reduce="min")

    assert torch.equal(summed, torch.tensor([[4.0, 6.0], [6.0, 0.5]], dtype=torch.float32))
    assert torch.equal(maxed, torch.tensor([[3.0, 4.0], [6.0, 0.5]], dtype=torch.float32))
    assert torch.equal(mined, torch.tensor([[1.0, 2.0], [6.0, 0.5]], dtype=torch.float32))


def test_scatter_softmax_and_log_softmax_normalize_within_groups() -> None:
    src = torch.tensor([[1.0, 0.0], [2.0, -1.0], [3.0, 5.0], [0.0, 1.0]], dtype=torch.float32)
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    softmax = scatter_softmax(src=src, index=index, dim=0)
    log_softmax = scatter_log_softmax(src=src, index=index, dim=0)

    expected_group0 = torch.softmax(src[:2], dim=0)
    expected_group1 = torch.softmax(src[2:], dim=0)
    expected = torch.cat([expected_group0, expected_group1], dim=0)

    assert torch.allclose(softmax, expected)
    assert torch.allclose(log_softmax.exp(), expected)
