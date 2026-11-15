# SPDX-License-Identifier: Apache-2.0
"""Validate worker transfer boundaries without allocating NPU memory."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
from lmcache.v1.multiprocess import xds_transfer as common_transfer
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.xds_catalog import XDSExtent, XDSLease
import pytest
import torch

# First Party
from lmcache_ascend.v1 import xds_transfer


@pytest.fixture
def transfer(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple:
    """Register a two-plane pool through the public worker API."""
    monkeypatch.setenv("LMCACHE_XDS_MLA_READ_MODE", "direct")
    client = Mock()
    client.get_blob_size.return_value = 65536
    client.get_io_unit_size.return_value = 4096
    client.get_blob.return_value = 1
    monkeypatch.setattr(common_transfer.xds, "XDSClient", Mock(return_value=client))
    settings = getattr(request, "param", 1)
    block_sizes = (
        settings.get("sizes", (4096, 4096))
        if isinstance(settings, dict)
        else (4096, 4096)
    )
    blocks_in_chunk = settings["blocks"] if isinstance(settings, dict) else settings
    pool = Mock(
        planes=(
            SimpleNamespace(buffer_ref=1, block_bytes=block_sizes[0], num_blocks=16),
            SimpleNamespace(buffer_ref=2, block_bytes=block_sizes[1], num_blocks=16),
        )
    )
    monkeypatch.setattr(xds_transfer, "RegisteredNPUKVPool", Mock(return_value=pool))
    rpc = Mock()
    monkeypatch.setattr(common_transfer, "xds_rpc", rpc)
    monkeypatch.setattr(torch.npu, "synchronize", Mock())
    monkeypatch.setenv("LMCACHE_XDS_BLOB_COUNT", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    tensor = Mock(ndim=4, shape=(16, 128, 1, 16), device=SimpleNamespace(index=0))
    caches = {"layer": (tensor, tensor)}
    replica = settings.get("replica", False) if isinstance(settings, dict) else False
    context = xds_transfer.XDSWorkerTransferContext(
        10,
        Mock(),
        0,
        reader_id=1 if replica else 0,
        io_world_size=4 if replica else None,
        num_readers=4 if replica else 1,
        writer=not replica,
    )
    groups = settings.get("groups", ()) if isinstance(settings, dict) else ()
    context.register(
        caches, "model", 1, blocks_in_chunk, 1.0, engine_group_infos=groups
    )
    rpc.reset_mock()
    return context, caches, client, rpc


def key(start: int = 0, end: int = 256) -> IPCCacheServerKey:
    """Construct a worker request for two complete chunks."""
    return IPCCacheServerKey(
        "model", 1, 0, tuple(range(256)), start, end, "req", num_kv_readers=1
    )


@pytest.mark.parametrize("lease_count", [0, 1, 3])
def test_bad_read_count_submits_no_dma(transfer: tuple, lease_count: int) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.return_value = tuple(
        XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(lease_count)
    )
    with pytest.raises(RuntimeError, match="Metadata chunk count"):
        context.submit_retrieve("req", key(), caches, [[0, 1]], None, 1)
    client.submit_readv_plan.assert_not_called()
    context.close()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "start,end", [(-128, 128), (0, 0), (128, 0), (0, 257), (1, 256), (0, 255)]
)
@pytest.mark.parametrize("operation", ["submit_store", "submit_retrieve"])
def test_bad_token_range_precedes_metadata(
    transfer: tuple, start: int, end: int, operation: str
) -> None:
    context, caches, client, rpc = transfer
    with pytest.raises(ValueError, match="complete aligned"):
        getattr(context, operation)("req", key(start, end), caches, [[0, 1]], None, 1)
    rpc.assert_not_called()
    client.submit_writev.assert_not_called()
    client.submit_readv_plan.assert_not_called()
    context.close()


@pytest.mark.parametrize("operation", ["submit_store", "submit_retrieve"])
def test_successful_transfer_drains_before_close(
    transfer: tuple, operation: str
) -> None:
    context, caches, client, rpc = transfer
    leases = tuple(XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(2))
    rpc.return_value.result.return_value = leases
    client.submit_writev.side_effect = [10, 11]
    client.submit_readv_plan.side_effect = [10, 11]
    future = getattr(context, operation)("req", key(), caches, [[5, 2]], None, 1)
    assert future.result() is True
    assert [call.args[0] for call in client.wait.call_args_list] == [10, 11]
    completion = rpc.call_args.args[2]
    assert completion.tickets == ("0", "1")
    context.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("operation", ["submit_store", "submit_retrieve"])
def test_timeout_retains_hbm_and_blobs(transfer: tuple, operation: str) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.return_value = tuple(
        XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(2)
    )
    client.wait.side_effect = TimeoutError("DMA still pending")
    with pytest.raises(TimeoutError):
        getattr(context, operation)("req", key(), caches, [[5, 2]], None, 1)
    with pytest.raises(RuntimeError, match="undrained"):
        context.recover_registration()
    with pytest.raises(RuntimeError, match="uncertain"):
        context.close()
    client.release_blob.assert_not_called()
    client.unregister_buffer.assert_not_called()
    client.close.assert_not_called()
    assert rpc.call_count == 1


def test_recovery_reuses_connection_and_hbm(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    assert context.recover_registration() is True
    assert context.recover_registration() is True
    client.connect.assert_called_once()
    client.get_blob.assert_called_once()
    assert [call.args[1] for call in rpc.call_args_list] == [
        "ensure_registered",
        "bind_worker",
        "ensure_registered",
        "bind_worker",
    ]
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        context.recover_registration()


def test_recovery_rpc_timeout_can_retry_without_reconnecting(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.side_effect = [
        TimeoutError("metadata unavailable"),
        None,
        None,
    ]
    with pytest.raises(TimeoutError):
        context.recover_registration()
    assert context.recover_registration() is True
    client.connect.assert_called_once()
    context.close()


def test_unregister_prevents_new_dma_and_recovery(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    context.unregister()
    with pytest.raises(RuntimeError, match="not ready"):
        context.submit_store("req", key(), caches, [[5, 2]], None, 1)
    with pytest.raises(RuntimeError, match="closed"):
        context.recover_registration()
    client.submit_writev.assert_not_called()
    context.close()


def test_recovery_waits_for_transfer_completion(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    entered = Event()
    release = Event()
    attempting_recovery = Event()
    calls = []
    leases = tuple(XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(2))

    def metadata(*args: object) -> Mock:
        calls.append(args[1])
        return Mock(result=Mock(return_value=leases))

    def wait(*args: object, **kwargs: object) -> None:
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release I/O")

    def recover() -> bool:
        attempting_recovery.set()
        return context.recover_registration()

    rpc.side_effect = metadata
    client.wait.side_effect = wait
    with ThreadPoolExecutor(max_workers=2) as executor:
        transfer_result = executor.submit(
            context.submit_store, "req", key(), caches, [[5, 2]], None, 1
        )
        try:
            assert entered.wait(5)
            recovery_result = executor.submit(recover)
            assert attempting_recovery.wait(5)
            assert calls == ["reserve_tokens"]
        finally:
            release.set()
        assert transfer_result.result(timeout=5).result() is True
        assert recovery_result.result(timeout=5) is True
    assert calls == [
        "reserve_tokens",
        "publish_many",
        "ensure_registered",
        "bind_worker",
    ]
    context.close()


@pytest.mark.parametrize("transfer", [2], indirect=True)
def test_partial_chunk_preserves_shared_prefix(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.return_value = (
        XDSLease("ticket", XDSExtent(1, 16384, 16384)),
    )
    result = context.submit_retrieve("req", key(), caches, [[5, 2]], None, 2, 128)
    assert result.result() is True
    assert [call.args for call in client.submit_readv.call_args_list] == [
        (1, 20480, [(1, 8192, 4096)]),
        (1, 28672, [(2, 8192, 4096)]),
    ]
    client.submit_readv_plan.assert_not_called()
    assert client.wait.call_count == 2
    assert rpc.call_args.args[2].tickets == ("ticket",)
    context.close()


@pytest.mark.parametrize("skip", [-128, 1, 127, 257, 384])
def test_invalid_skip_does_not_claim_or_write(transfer: tuple, skip: int) -> None:
    context, caches, client, rpc = transfer
    with pytest.raises(ValueError, match="Skipped prefix"):
        context.submit_retrieve("req", key(), caches, [[5, 2]], None, 1, skip)
    rpc.assert_not_called()
    client.submit_readv.assert_not_called()
    client.submit_readv_plan.assert_not_called()
    context.close()


def test_skip_whole_chunks_releases_all_claims_without_overwriting(
    transfer: tuple,
) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.return_value = tuple(
        XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(2)
    )
    result = context.submit_retrieve("req", key(), caches, [[5, 2]], None, 1, 256)
    assert result.result() is True
    client.submit_readv.assert_not_called()
    client.submit_readv_plan.assert_not_called()
    client.wait.assert_not_called()
    assert rpc.call_args.args[2].tickets == ("0", "1")
    context.close()


@pytest.mark.parametrize("transfer", [3], indirect=True)
@pytest.mark.parametrize("skip_blocks", [0, 1, 2, 3, 4, 6])
def test_noncontiguous_restore_preserves_every_skipped_byte(
    transfer: tuple, skip_blocks: int
) -> None:
    context, caches, client, rpc = transfer
    block_bytes = 4096
    blocks = [9, 1, 11, 3, 14, 5]
    memory = {ref: bytearray([0xA5]) * (16 * block_bytes) for ref in (1, 2)}
    expected = {ref: data.copy() for ref, data in memory.items()}
    disk = bytearray()
    for chunk in range(2):
        for ref in (1, 2):
            for local in range(3):
                logical = chunk * 3 + local
                payload = bytes([ref * 32 + logical]) * block_bytes
                disk.extend(payload)
                if logical >= skip_blocks:
                    offset = blocks[logical] * block_bytes
                    expected[ref][offset : offset + block_bytes] = payload

    def read(blob: int, offset: int, slices: list) -> int:
        for ref, destination, size in slices:
            memory[ref][destination : destination + size] = disk[offset : offset + size]
            offset += size
        return 1

    def plan_read(blob: int, offset: int, plan: tuple) -> int:
        ids, groups, size = plan
        return read(
            blob,
            offset,
            [(ref, block * size, size) for ref, _, _ in groups for block in ids],
        )

    client.prepare_indexed_readv.side_effect = lambda *args: args
    client.submit_readv_plan.side_effect = plan_read
    client.submit_readv.side_effect = read
    rpc.return_value.result.return_value = tuple(
        XDSLease(str(i), XDSExtent(1, i * 24576, 24576)) for i in range(2)
    )
    request = IPCCacheServerKey(
        "model", 1, 0, tuple(range(768)), 0, 768, "req", num_kv_readers=1
    )
    assert (
        context.submit_retrieve(
            "req", request, caches, [blocks], None, 3, skip_blocks * 128
        ).result()
        is True
    )
    assert memory == expected
    context.close()


def test_retrieve_claim_payload_contains_only_range_identity(transfer: tuple) -> None:
    context, caches, client, rpc = transfer
    rpc.return_value.result.return_value = (XDSLease("read", XDSExtent(1, 8192, 8192)),)
    context.submit_retrieve("req", key(128, 256), caches, [[3]], None, 1).result()
    claim = rpc.call_args_list[0]
    assert claim.args[1] == "claim_range"
    request = claim.args[2]
    assert (request.lookup, request.worker, request.worker_id) == ("req", 10, 0)
    assert (request.model_name, request.world_size, request.start, request.end) == (
        "model",
        1,
        128,
        256,
    )
    assert not hasattr(request, "token_ids")
    assert not hasattr(request, "key")
    context.close()


@pytest.mark.parametrize(
    "transfer", [{"blocks": 2, "sizes": (4096, 8192)}], indirect=True
)
@pytest.mark.parametrize("skip_blocks", [0, 1, 2, 3, 4])
def test_unequal_mla_planes_preserve_skipped_pages(
    transfer: tuple, skip_blocks: int
) -> None:
    context, caches, client, rpc = transfer
    blocks = [7, 2, 11, 4]
    sizes = {1: 4096, 2: 8192}
    memory = {ref: bytearray([0xAA]) * (16 * size) for ref, size in sizes.items()}
    expected = {ref: bytearray(data) for ref, data in memory.items()}
    disk = bytearray()
    for chunk in range(2):
        for ref, size in sizes.items():
            for within in range(2):
                logical = chunk * 2 + within
                data = bytes([ref * 16 + logical]) * size
                disk.extend(data)
                if logical >= skip_blocks:
                    offset = blocks[logical] * size
                    expected[ref][offset : offset + size] = data

    def read(blob: int, offset: int, slices: list) -> int:
        for ref, destination, size in slices:
            memory[ref][destination : destination + size] = disk[offset : offset + size]
            offset += size
        return 1

    client.submit_readv.side_effect = read
    rpc.return_value.result.return_value = tuple(
        XDSLease(str(i), XDSExtent(1, i * 24576, 24576)) for i in range(2)
    )
    request = IPCCacheServerKey(
        "model", 1, 0, tuple(range(512)), 0, 512, "req", num_kv_readers=1
    )
    context.submit_retrieve(
        "req", request, caches, [blocks], None, 2, skip_blocks * 128
    ).result()
    assert memory == expected
    client.submit_readv_plan.assert_not_called()
    context.close()


@pytest.mark.parametrize("transfer", [{"blocks": 1, "replica": True}], indirect=True)
@pytest.mark.parametrize("timeout", [False, True])
def test_replica_imports_survive_until_dma_drains(
    transfer: tuple, timeout: bool
) -> None:
    context, caches, client, rpc = transfer
    timeline = Mock()
    timeline.attach_mock(client, "client")
    timeline.attach_mock(rpc, "rpc")
    rpc.return_value.result.return_value = (
        XDSLease("shared", XDSExtent(55, 0, 8192, 123)),
    )
    request = IPCCacheServerKey(
        "model", 1, 0, tuple(range(128)), 0, 128, "req", num_kv_readers=4
    )
    client.get_blob.assert_not_called()
    if timeout:
        client.wait.side_effect = TimeoutError("DMA pending")
        with pytest.raises(TimeoutError):
            context.submit_retrieve("req", request, caches, [[3]], None, 1)
        client.unimport_blob_read.assert_not_called()
        with pytest.raises(RuntimeError, match="uncertain"):
            context.close()
        client.close.assert_not_called()
    else:
        context.submit_retrieve("req", request, caches, [[3]], None, 1).result()
        client.import_blob_read.assert_called_once_with(55, 123)
        client.unimport_blob_read.assert_called_once_with(55)
        names = [call[0] for call in timeline.mock_calls]
        assert names.index("client.import_blob_read") < names.index("client.wait")
        assert names.index("client.wait") < names.index("client.unimport_blob_read")
        context.close()
        client.release_blob.assert_not_called()


@pytest.mark.parametrize("widths", [(576,), (128,), (128, 16)])
def test_dsa_planes_retained_without_payload_copy(
    transfer: tuple, widths: tuple[int, ...]
) -> None:
    context, _, _, _ = transfer
    tensors = tuple(torch.empty((16, 128, 1, width), device="cpu") for width in widths)
    assert context.prepare_planes({"indexer": tensors}) == tensors
    context.close()


@pytest.mark.parametrize(
    "transfer",
    [
        {
            "blocks": 1,
            "groups": (EngineGroupInfo(0, (0,), 128), EngineGroupInfo(0, (1,), 128)),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("operation", ["submit_store", "submit_retrieve"])
def test_physical_groups_use_one_shared_block_map(
    transfer: tuple, operation: str
) -> None:
    context, caches, client, rpc = transfer
    leases = tuple(XDSLease(str(i), XDSExtent(1, i * 8192, 8192)) for i in range(2))
    rpc.return_value.result.return_value = leases
    future = getattr(context, operation)(
        "req", key(), caches, [[5, 2], [5, 2]], None, 1
    )
    assert future.result() is True
    assert client.wait.call_count == 2
    client.reset_mock()
    rpc.reset_mock()
    with pytest.raises(ValueError, match="group geometry"):
        getattr(context, operation)("req", key(), caches, [[5, 2], [2, 5]], None, 1)
    rpc.assert_not_called()
    client.submit_writev.assert_not_called()
    client.submit_readv.assert_not_called()
    context.close()
