# SPDX-License-Identifier: Apache-2.0
"""Worker-local fixed HBM registrations for xDS multiprocess transfers."""

# Third Party
from lmcache.v1.multiprocess.xds_cuda_pool import (
    RegisteredKVPlane,
    RegistrationClient,
)
import torch


def get_npu_readv_registration(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the owned, padded 2 MiB range required by Ascend HAL.

    This is the fixed-pool validation from the previous xDS integration. It
    runs once at pool initialization, not once per request. Padding must be
    inside the tensor's owning storage; rounding outside that storage could
    register another allocation. Logical I/O never includes padding.

    Args:
        tensor: Contiguous, block-major NPU KV plane with an aligned base.

    Returns:
        Registration address and rounded byte length.

    Raises:
        ValueError: If the plane is empty, non-NPU, not contiguous, unaligned,
            or lacks owned padding for its final HAL page.
    """
    alignment = 2 * 1024 * 1024
    pointer = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    registered_size = (size + alignment - 1) // alignment * alignment
    storage = tensor.untyped_storage()
    if (
        tensor.device.type != "npu"
        or tensor.ndim < 2
        or not tensor.is_contiguous()
        or pointer % alignment
        or registered_size <= 0
        or pointer < storage.data_ptr()
        or pointer + registered_size > storage.data_ptr() + storage.nbytes()
    ):
        raise ValueError(
            "xDS needs a contiguous, owned, padded 2 MiB aligned NPU KV plane; "
            f"device={tensor.device}, shape={tuple(tensor.shape)}, "
            f"contiguous={tensor.is_contiguous()}, "
            f"alignment_remainder={pointer % alignment}, "
            f"required_end={pointer - storage.data_ptr() + registered_size}, "
            f"storage_bytes={storage.nbytes()}"
        )
    return pointer, registered_size


class RegisteredNPUKVPool:
    """Retain final KV tensors and register each distinct range exactly once.

    The caller initializes its NPU before creating this pool and owns the xDS
    connection. No payload buffers are allocated. Tensors must not be resized
    or rebound while the pool is open. The caller must drain all transfers
    before close(), and close the pool before freeing vLLM's KV allocation.
    """

    def __init__(
        self, client: RegistrationClient, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        if not tensors:
            raise ValueError("KV pool cannot be empty")
        ranges = tuple(get_npu_readv_registration(tensor) for tensor in tensors)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("Each worker KV pool must belong to one NPU")
        self._client = client
        self._tensors = tensors
        self._registrations: dict[tuple[int, int], int] = {}
        planes = []
        try:
            for tensor, region in zip(tensors, ranges, strict=True):
                if region not in self._registrations:
                    self._registrations[region] = client.register_gpu_buffer(*region)
                planes.append(
                    RegisteredKVPlane(
                        self._registrations[region],
                        tensor.stride(0) * tensor.element_size(),
                        tensor.shape[0],
                    )
                )
        except (RuntimeError, OSError):
            self.close()
            raise
        self.planes = tuple(planes)

    def close(self) -> None:
        """Unregister a drained pool, then release its retained tensors.

        Repeated successful closes are no-ops. If unregister raises, remaining
        handles and tensors are retained so cleanup can be retried without
        freeing memory still registered for DMA.
        """
        for region, handle in tuple(self._registrations.items()):
            self._client.unregister_buffer(handle)
            del self._registrations[region]
        self._tensors = ()
