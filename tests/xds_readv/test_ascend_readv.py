# SPDX-License-Identifier: Apache-2.0
"""Ascend planner contract: supported layouts, exact offsets and safe rejection."""

import lmcache_ascend  # noqa: F401
import pytest
import torch
from lmcache.v1.memory_management import MemoryFormat
from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemNPUConnectorV2


def aligned_npu(shape, zero: bool = False):
    size = torch.Size(shape).numel() * 2
    owner = torch.empty(
        max(size, 16 * 2**20) + 2 * 2**20, device="npu", dtype=torch.uint8
    )
    offset = (-owner.data_ptr()) % (2 * 2**20)
    result = owner[offset : offset + size].view(torch.float16).reshape(shape)
    return result.zero_() if zero else result


@pytest.mark.parametrize("layout", ["SEPARATE", "2NB", "NB2"])
def test_readv_preserves_source_order_and_guards(layout: str) -> None:
    connector = VLLMPagedMemNPUConnectorV2(256, 3)
    if layout == "SEPARATE":
        caches = [
            (aligned_npu((8, 16, 2, 128), True), aligned_npu((8, 16, 2, 128), True))
            for _ in range(3)
        ]
        tensors = [t for layer in caches for t in layer]
    else:
        shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
        caches = [aligned_npu(shape, True) for _ in range(3)]
        tensors = caches
    slots = torch.tensor(list(range(64, 80)) + list(range(16, 32)))
    shape = torch.Size([2, 3, 32, 256])
    plan = connector.plan_xds_readv(
        [shape],
        [torch.float16],
        MemoryFormat.KV_2LTD,
        0,
        32,
        4096,
        kvcaches=caches,
        slot_mapping_cpu=slots,
    )
    assert plan is not None
    assert plan.registration_ranges == tuple(
        connector.get_xds_readv_registration(tensor) for tensor in tensors
    )
    source = (
        torch.arange(shape.numel(), dtype=torch.int32)
        .remainder(997)
        .to(torch.float16)
        .reshape(shape)
    )
    cursor = 0
    for index, offset, length in plan.slices:
        tensors[index].view(torch.uint8).flatten()[offset : offset + length].copy_(
            source.view(torch.uint8).flatten()[cursor : cursor + length]
        )
        cursor += length
    assert cursor == plan.size
    for layer in range(3):
        for kv in range(2):
            actual = (
                caches[layer][kv]
                if layout in ("SEPARATE", "2NB")
                else caches[layer][:, kv]
            ).cpu()
            expected = torch.zeros_like(actual)
            expected[4] = source[kv, layer, :16].reshape(16, 2, 128)
            expected[1] = source[kv, layer, 16:].reshape(16, 2, 128)
            assert torch.equal(actual, expected)


def test_batch_plans_carry_fresh_validated_ranges_and_slot_ownership() -> None:
    """Registration reuse within one restore must not reuse old block mappings."""
    connector = VLLMPagedMemNPUConnectorV2(256, 2)
    caches = [
        (aligned_npu((8, 16, 2, 128)), aligned_npu((8, 16, 2, 128))) for _ in range(2)
    ]
    chunks = [
        (
            [torch.Size([2, 2, 16, 256])],
            [torch.float16],
            MemoryFormat.KV_2LTD,
            s,
            s + 16,
        )
        for s in (0, 16)
    ]
    ranges = None
    for ids in ((4, 1), (3, 6)):
        slots = torch.tensor([b * 16 + t for b in ids for t in range(16)])
        plans = connector.plan_xds_readv_batch(
            chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
        )
        assert plans is not None
        assert tuple(plan.block_ids for plan in plans) == tuple((b,) for b in ids)
        assert plans[0].registration_ranges is plans[1].registration_ranges
        assert plans[0].registration_ranges is not ranges
        ranges = plans[0].registration_ranges
        assert ranges == tuple(
            connector.get_xds_readv_registration(tensor)
            for layer in caches
            for tensor in layer
        )


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "misaligned",
        "negative",
        "duplicate",
        "alias",
        "dtype",
        "noncontiguous",
        "prefix",
        "small_block",
        "ambiguous",
        "target_alignment",
        "mla",
    ],
)
def test_unsupported_returns_none_before_dma(case: str) -> None:
    connector = VLLMPagedMemNPUConnectorV2(256, 2)
    caches = [
        (aligned_npu((8, 16, 2, 128)), aligned_npu((8, 16, 2, 128))) for _ in range(2)
    ]
    slots = torch.arange(32)
    shape = torch.Size([2, 2, 32, 256])
    dtype = torch.float16
    kwargs = {}
    if case == "partial":
        slots = slots[:31]
        shape = torch.Size([2, 2, 31, 256])
    elif case == "misaligned":
        slots += 1
    elif case == "negative":
        slots[0] = -1
    elif case == "duplicate":
        slots[16:] = slots[:16]
    elif case == "alias":
        caches[1] = caches[0]
    elif case == "dtype":
        dtype = torch.float32
    elif case == "noncontiguous":
        caches[0] = tuple(t.transpose(1, 2) for t in caches[0])
    elif case == "prefix":
        kwargs["vllm_cached_tokens"] = 1
    elif case == "small_block":
        caches = [
            (aligned_npu((8, 16, 1, 8)), aligned_npu((8, 16, 1, 8))) for _ in range(2)
        ]
        shape = torch.Size([2, 2, 32, 8])
    elif case == "ambiguous":
        caches = [aligned_npu((2, 2, 16, 2, 128)) for _ in range(2)]
    elif case == "target_alignment":
        owner = aligned_npu((8, 16, 2, 128)).flatten()
        caches[0] = (owner[1:].reshape(1, 1, 1, -1), caches[0][1])
    elif case == "mla":
        connector.use_mla = True
    assert (
        connector.plan_xds_readv(
            [shape],
            [dtype],
            MemoryFormat.KV_2LTD,
            0,
            slots.numel(),
            4096,
            kvcaches=caches,
            slot_mapping_cpu=slots,
            **kwargs,
        )
        is None
    )


@pytest.mark.parametrize("layout", ["SEPARATE", "2NB", "NB2"])
def test_initialized_pool_reuses_ranges_and_handles_without_storage_queries(
    layout: str, monkeypatch
) -> None:
    """Fixed registrations survive changing request slots, but reject new pools."""
    connector = VLLMPagedMemNPUConnectorV2(256, 2)
    if layout == "SEPARATE":
        caches = [
            (aligned_npu((8, 16, 2, 128)), aligned_npu((8, 16, 2, 128)))
            for _ in range(2)
        ]
    else:
        shape = (2, 8, 16, 2, 128) if layout == "2NB" else (8, 2, 16, 2, 128)
        caches = [aligned_npu(shape) for _ in range(2)]
    pool = connector.initialize_xds_readv_pool(caches)
    pool.buffer_refs = tuple(range(101, 101 + len(pool.tensors)))

    def unexpected_query(tensor):
        raise AssertionError("restore queried registration range")

    monkeypatch.setattr(connector, "get_xds_readv_registration", unexpected_query)
    assert connector.initialize_xds_readv_pool(caches) is pool
    chunks = [
        ([torch.Size([2, 2, 16, 256])], [torch.float16], MemoryFormat.KV_2LTD, 0, 16)
    ]
    for block in (4, 1):
        plans = connector.plan_xds_readv_batch(
            chunks,
            4096,
            kvcaches=caches,
            slot_mapping_cpu=torch.arange(block * 16, (block + 1) * 16),
        )
        assert plans is not None and plans[0].block_ids == (block,)
        assert plans[0].registration_ranges is pool.registration_ranges
        assert plans[0].buffer_refs is pool.buffer_refs
    changed = list(caches)
    changed[0] = caches[1]
    with pytest.raises(RuntimeError, match="replacement"):
        connector.get_xds_readv_pool(changed)
    pool.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        connector.get_xds_readv_pool(caches)


@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
def test_readv_selected_spans_ignore_gaps_and_preserve_chunk_order(slot_dtype):
    """Unselected slots may be invalid; selected blocks must remain unique."""
    connector = VLLMPagedMemNPUConnectorV2(256, 2)
    caches = [
        (aligned_npu((8, 16, 2, 128)), aligned_npu((8, 16, 2, 128))) for _ in range(2)
    ]
    pool = connector.initialize_xds_readv_pool(caches)
    pool.buffer_refs = (101, 102, 103, 104)
    chunks = [
        (
            [torch.Size([2, 2, 16, 256])],
            [torch.float16],
            MemoryFormat.KV_2LTD,
            s,
            s + 16,
        )
        for s in (32, 0)
    ]
    # A strided CPU view also has to produce the exact logical block mapping.
    backing = torch.full((96,), -999, dtype=slot_dtype)
    slots = backing[::2]
    slots[:16] = torch.arange(16, 32, dtype=slot_dtype)
    slots[32:] = torch.arange(64, 80, dtype=slot_dtype)
    plans = connector.plan_xds_readv_batch(
        chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
    )
    assert plans is not None
    assert [p.block_ids for p in plans] == [(4,), (1,)]
    assert all(p.buffer_refs is pool.buffer_refs for p in plans)
    assert plans[0].bind_slices(pool.buffer_refs) == [
        (ref, 4 * 8192, 8192) for ref in (101, 103, 102, 104)
    ]
    # Duplicate physical targets across nonadjacent source spans are rejected.
    slots[32:] = slots[:16]
    assert (
        connector.plan_xds_readv_batch(
            chunks, 4096, kvcaches=caches, slot_mapping_cpu=slots
        )
        is None
    )


def test_readv_rejects_incomplete_registered_references_before_submission():
    """An initialized pool must cover every K/V target before any DMA."""
    connector = VLLMPagedMemNPUConnectorV2(256, 1)
    caches = [(aligned_npu((8, 16, 2, 128)), aligned_npu((8, 16, 2, 128)))]
    pool = connector.initialize_xds_readv_pool(caches)
    pool.buffer_refs = (101,)
    chunks = [
        ([torch.Size([2, 1, 16, 256])], [torch.float16], MemoryFormat.KV_2LTD, 0, 16)
    ]
    with pytest.raises(ValueError, match="reference count"):
        connector.plan_xds_readv_batch(
            chunks, 4096, kvcaches=caches, slot_mapping_cpu=torch.arange(16)
        )
