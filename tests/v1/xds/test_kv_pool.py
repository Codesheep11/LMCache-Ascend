# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache_ascend.v1.xds_kv_pool import (
    RegisteredNPUKVPool,
    get_npu_readv_registration,
)

ALIGNMENT = 2 * 1024 * 1024
DEVICE = Mock(type="npu")


def plane(pointer: int, size: int = 4096, storage_size: int = ALIGNMENT) -> Mock:
    """Represent an owned aligned device plane without allocating hardware."""
    tensor = Mock(device=DEVICE, ndim=2, shape=(1, size))
    tensor.data_ptr.return_value = pointer
    tensor.numel.return_value = size
    tensor.element_size.return_value = 1
    tensor.stride.return_value = size
    tensor.is_contiguous.return_value = True
    storage = tensor.untyped_storage.return_value
    storage.data_ptr.return_value = pointer
    storage.nbytes.return_value = storage_size
    return tensor


def test_register_once_for_aliases_and_close_once() -> None:
    client = Mock()
    client.register_gpu_buffer.return_value = 42
    tensor = plane(ALIGNMENT)
    pool = RegisteredNPUKVPool(client, (tensor, tensor))
    client.register_gpu_buffer.assert_called_once_with(ALIGNMENT, ALIGNMENT)
    assert [p.buffer_ref for p in pool.planes] == [42, 42]
    assert pool.planes[0].block_bytes == 4096
    pool.close()
    pool.close()
    client.unregister_buffer.assert_called_once_with(42)


def test_registration_failure_rolls_back_successful_ranges() -> None:
    client = Mock()
    client.register_gpu_buffer.side_effect = [42, RuntimeError("HAL failure")]
    with pytest.raises(RuntimeError, match="HAL failure"):
        RegisteredNPUKVPool(client, (plane(ALIGNMENT), plane(2 * ALIGNMENT)))
    client.unregister_buffer.assert_called_once_with(42)


def test_close_failure_can_be_retried() -> None:
    client = Mock()
    client.register_gpu_buffer.return_value = 42
    client.unregister_buffer.side_effect = [RuntimeError("busy"), None]
    pool = RegisteredNPUKVPool(client, (plane(ALIGNMENT),))
    with pytest.raises(RuntimeError, match="busy"):
        pool.close()
    pool.close()
    assert client.unregister_buffer.call_count == 2


@pytest.mark.parametrize(
    "pointer,size,storage", [(ALIGNMENT + 1, 4096, ALIGNMENT), (ALIGNMENT, 4096, 4096)]
)
def test_reject_unowned_hal_page(pointer: int, size: int, storage: int) -> None:
    client = Mock()
    with pytest.raises(ValueError, match="owned"):
        RegisteredNPUKVPool(client, (plane(pointer, size, storage),))
    client.register_gpu_buffer.assert_not_called()


def test_reject_cpu_and_noncontiguous_planes() -> None:
    with pytest.raises(ValueError):
        get_npu_readv_registration(torch.empty(2, 2))
    tensor = plane(ALIGNMENT)
    tensor.is_contiguous.return_value = False
    with pytest.raises(ValueError):
        get_npu_readv_registration(tensor)
