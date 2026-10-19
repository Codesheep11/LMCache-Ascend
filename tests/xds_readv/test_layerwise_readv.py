# SPDX-License-Identifier: Apache-2.0
"""Layerwise Ascend plans preserve KV bytes, block ownership and store format."""

# Standard
from types import SimpleNamespace

# Third Party
import lmcache_ascend  # noqa: F401
import pytest
import torch
from lmcache.v1.memory_management import MemoryFormat

# First Party
from lmcache_ascend.v1.npu_connector.xds_layerwise import (
    XDSReadvLayerwiseNPUConnector,
)


def aligned_npu(shape: tuple[int, ...]) -> torch.Tensor:
    """Return a padded NPU KV view satisfying the HAL registration contract."""
    size = torch.Size(shape).numel() * 2
    owner = torch.empty(16 * 2**20 + 2 * 2**20, dtype=torch.uint8, device="npu")
    offset = (-owner.data_ptr()) % (2 * 2**20)
    return owner[offset : offset + size].view(torch.bfloat16).reshape(shape).fill_(-7)


def fixture(layout: str) -> tuple[XDSReadvLayerwiseNPUConnector, list, torch.Tensor]:
    """Build three layers and a fragmented two-block slot assignment."""
    connector = XDSReadvLayerwiseNPUConnector(
        128,
        3,
        False,
        chunk_size=128,
        dtype=torch.bfloat16,
        device=torch.device("npu:0"),
    )
    if layout == "SEPARATE":
        caches = [
            (aligned_npu((8, 128, 1, 128)), aligned_npu((8, 128, 1, 128)))
            for _ in range(3)
        ]
    else:
        shape = (2, 8, 128, 1, 128) if layout == "2NB" else (8, 2, 128, 1, 128)
        caches = [aligned_npu(shape) for _ in range(3)]
    slots = torch.tensor([b * 128 + t for b in (5, 1) for t in range(128)])
    return connector, caches, slots


@pytest.mark.parametrize("layout", ["SEPARATE", "2NB", "NB2"])
def test_layerwise_plans_preserve_bytes_and_untouched_blocks(layout: str) -> None:
    connector, caches, slots = fixture(layout)
    chunks = [
        ([torch.Size([2, 128, 128])], [torch.bfloat16], MemoryFormat.KV_2TD, s, s + 128)
        for s in (0, 128)
    ]
    plans = connector.plan_layerwise_xds_readv(
        chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
    )
    assert plans is not None and len(plans) == 3
    for layer, layer_plans in enumerate(plans):
        for chunk, plan in enumerate(layer_plans):
            assert plan.size == 65536 and plan.registration_ranges is not None
            source = (
                (
                    torch.arange(2 * 128 * 128, dtype=torch.float32).remainder(997) / 32
                    + layer
                    + chunk
                )
                .to(torch.bfloat16)
                .reshape(2, 128, 128)
            )
            cursor = 0
            for index, offset, length in plan.slices:
                target = plan.tensors[index].view(torch.uint8).flatten()
                target[offset : offset + length].copy_(
                    source.view(torch.uint8).flatten()[cursor : cursor + length]
                )
                cursor += length
            assert cursor == source.numel() * 2
            for kv in range(2):
                actual = (
                    caches[layer][kv]
                    if layout in ("SEPARATE", "2NB")
                    else caches[layer][:, kv]
                ).cpu()
                assert torch.equal(actual[(5, 1)[chunk]].reshape(128, 128), source[kv])
        for kv in range(2):
            actual = (
                caches[layer][kv]
                if layout in ("SEPARATE", "2NB")
                else caches[layer][:, kv]
            ).cpu()
            assert all(torch.all(actual[b] == -7) for b in (0, 2, 3, 4, 6, 7))
    fresh = connector.plan_layerwise_xds_readv(
        chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots.flip(0)
    )
    assert fresh is None  # Reversed token order is not a full contiguous KV block.


@pytest.mark.parametrize("layout", ["SEPARATE", "2NB", "NB2"])
def test_layerwise_store_kernel_emits_explicit_kv_major_bytes(layout: str) -> None:
    connector, caches, slots = fixture(layout)
    expected = []
    for layer in range(3):
        source = (
            (
                torch.arange(2 * 256 * 128, dtype=torch.float32).remainder(997) / 32
                + layer
            )
            .to(torch.bfloat16)
            .reshape(2, 256, 128)
        )
        expected.append(source)
        for kv in range(2):
            target = (
                caches[layer][kv]
                if layout in ("SEPARATE", "2NB")
                else caches[layer][:, kv]
            )
            for chunk, block in enumerate((5, 1)):
                target[block].copy_(
                    source[kv, chunk * 128 : (chunk + 1) * 128].reshape(128, 1, 128)
                )
    objects = [
        [
            SimpleNamespace(
                tensor=torch.empty((2, 128, 128), dtype=torch.bfloat16, device="npu"),
                metadata=SimpleNamespace(fmt=MemoryFormat.KV_T2D),
            )
            for _ in range(2)
        ]
        for _ in range(3)
    ]
    iterator = connector.batched_from_gpu(
        objects,
        [0, 128],
        [128, 256],
        kvcaches=caches,
        slot_mapping=slots.to("npu"),
        sync=True,
    )
    assert len(list(iterator)) == 4
    torch.npu.synchronize()
    for layer in range(3):
        for chunk, obj in enumerate(objects[layer]):
            assert obj.metadata.fmt == MemoryFormat.KV_2TD
            assert torch.equal(
                obj.tensor.cpu(), expected[layer][:, chunk * 128 : (chunk + 1) * 128]
            )


@pytest.mark.parametrize("bad", ["legacy", "partial", "overlap"])
def test_layerwise_rejects_unsupported_spans_before_dma(bad: str) -> None:
    connector, caches, slots = fixture("SEPARATE")
    chunks = [
        ([torch.Size([2, 128, 128])], [torch.bfloat16], MemoryFormat.KV_2TD, s, s + 128)
        for s in (0, 128)
    ]
    if bad == "legacy":
        chunks = [
            (shape, dtype, MemoryFormat.KV_T2D, a, b)
            for shape, dtype, _, a, b in chunks
        ]
    elif bad == "partial":
        slots[0] = slots[0] + 1
    else:
        slots[128:] = slots[:128]
    assert (
        connector.plan_layerwise_xds_readv(
            chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
        )
        is None
    )


@pytest.mark.parametrize("layout", ["SEPARATE", "2NB", "NB2"])
def test_layerwise_initialized_pool_binds_correct_layer_handles(
    layout: str, monkeypatch
) -> None:
    """Each layer uses its startup handles without repeating HAL range queries."""
    connector, caches, slots = fixture(layout)
    pool = connector.initialize_xds_readv_pool(caches)
    pool.buffer_refs = tuple(range(101, 101 + len(pool.tensors)))

    def unexpected_query(tensor):
        raise AssertionError("restore queried registration range")

    monkeypatch.setattr(
        "lmcache_ascend.v1.npu_connector.xds_layerwise.get_npu_readv_registration",
        unexpected_query,
    )
    chunks = [
        ([torch.Size([2, 128, 128])], [torch.bfloat16], MemoryFormat.KV_2TD, s, s + 128)
        for s in (0, 128)
    ]
    plans = connector.plan_layerwise_xds_readv(
        chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
    )
    assert plans is not None
    for layer, layer_plans in enumerate(plans):
        refs = (
            pool.buffer_refs[2 * layer : 2 * layer + 2]
            if layout == "SEPARATE"
            else (pool.buffer_refs[layer],)
        )
        assert all(plan.buffer_refs == refs for plan in layer_plans)
