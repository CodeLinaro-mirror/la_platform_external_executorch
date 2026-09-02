#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for the MLX partitioner.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.test.test_utils import get_mlx_node_counts
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from torch.export import export


class TestMLXPartitionerRejectsToEdge(unittest.TestCase):
    """MLXPartitioner must only be used via to_edge_transform_and_lower."""

    def test_to_edge_then_to_backend_raises(self):
        class M(nn.Module):
            def forward(self, x):
                return x + 1

        ep = export(M(), (torch.randn(4),), strict=False)
        edge = to_edge(
            ep,
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
        )

        with self.assertRaises(RuntimeError) as ctx:
            edge.to_backend(MLXPartitioner())

        self.assertIn("to_edge_transform_and_lower", str(ctx.exception))


def _lower(model, inputs):
    return to_edge_transform_and_lower(
        export(model, inputs, strict=False),
        partitioner=[MLXPartitioner()],
    ).to_executorch()


def _node_counts(model, inputs):
    lowered = _lower(model, inputs)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pte"
        path.write_bytes(lowered.buffer)
        return get_mlx_node_counts(path)


class Sdpa(nn.Module):
    def __init__(self, is_causal: bool = False):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal
        )


class TestMLXPartitionerSdpaShapes(unittest.TestCase):
    """The fused attention kernel takes rank 4, so other ranks are adapted or left."""

    def test_rank4_is_unchanged(self):
        counts = _node_counts(
            Sdpa(), tuple(torch.randn(2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 0)

    def test_rank3_is_lifted_and_still_fused(self):
        counts = _node_counts(Sdpa(), tuple(torch.randn(2, 16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertEqual(counts.get("SqueezeNode", 0), 1)

    def test_rank2_is_lifted_twice(self):
        counts = _node_counts(Sdpa(), tuple(torch.randn(16, 64) for _ in range(3)))
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 6)

    def test_rank5_is_not_claimed(self):
        # Folding the leading dimensions would change which of them the kernel reads
        # as heads, so this is left to decompose.
        counts = _node_counts(
            Sdpa(), tuple(torch.randn(2, 2, 4, 16, 64) for _ in range(3))
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)

    def test_unequal_batch_is_not_claimed(self):
        counts = _node_counts(
            Sdpa(),
            (
                torch.randn(2, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
                torch.randn(1, 4, 16, 64),
            ),
        )
        self.assertEqual(counts.get("SdpaNode", 0), 0)


class TestMLXPartitionerSdpaCausal(unittest.TestCase):
    """Causal attention is unchanged by this file's other guards."""

    def _run(self, inputs, is_causal):
        model = Sdpa(is_causal).eval()
        with torch.no_grad():
            ref = model(*inputs)
        lowered = _lower(model, inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pte"
            path.write_bytes(lowered.buffer)
            counts = get_mlx_node_counts(path)
            from executorch.runtime import Runtime

            method = Runtime.get().load_program(path).load_method("forward")
            out = method.execute(list(inputs))[0]
        return counts, (out - ref).abs().max().item()

    def test_equal_lengths_stay_fused(self):
        counts, err = self._run(
            tuple(torch.randn(1, 4, 16, 64) for _ in range(3)), is_causal=True
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertLess(err, 1e-4)

    def test_rank3_causal_is_lifted(self):
        counts, err = self._run(
            tuple(torch.randn(2, 16, 64) for _ in range(3)), is_causal=True
        )
        self.assertEqual(counts.get("SdpaNode", 0), 1)
        self.assertEqual(counts.get("ExpandDimsNode", 0), 3)
        self.assertLess(err, 1e-4)


class TestMLXPartitionerMixedSupport(unittest.TestCase):
    """An operator is preserved from decomposition per operator, not per call."""

    def test_supported_and_unsupported_calls_in_one_graph(self):
        class Mixed(nn.Module):
            def forward(self, a, b, c, d):
                # One call the kernel takes, one it does not.
                x = torch.nn.functional.scaled_dot_product_attention(a, a, a)
                y = torch.nn.functional.scaled_dot_product_attention(b, c, d)
                return x.sum() + y.sum()

        inputs = (
            torch.randn(1, 4, 16, 64),
            torch.randn(2, 4, 16, 64),
            torch.randn(1, 4, 16, 64),
            torch.randn(1, 4, 16, 64),
        )
        # Without giving the whole operator back, the unsupported call would be
        # neither lowered nor decomposed and this would raise.
        _lower(Mixed().eval(), inputs)

    def test_mixed_support_outside_attention(self):
        class TwoRolls(nn.Module):
            def forward(self, x):
                return torch.roll(x, 1, dims=0).sum() + torch.roll(x, 1).sum()

        inputs = (torch.randn(4, 8),)
        model = TwoRolls().eval()
        with torch.no_grad():
            ref = model(*inputs)
        lowered = _lower(model, inputs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pte"
            path.write_bytes(lowered.buffer)
            from executorch.runtime import Runtime

            method = Runtime.get().load_program(path).load_method("forward")
            out = method.execute(list(inputs))[0]
        self.assertLess((out - ref).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
