# SPDX-License-Identifier: Apache-2.0
"""Ascend final-page registration and device fencing for shared xDS transfers."""

# Standard
import os

# Third Party
from lmcache.v1.multiprocess.xds_transfer import (
    XDSWorkerTransferContext as BaseXDSContext,
)
import torch

# First Party
from lmcache_ascend.v1.xds_kv_pool import RegisteredNPUKVPool


class XDSWorkerTransferContext(BaseXDSContext):
    """Use the common MLA leader/broadcast pipeline with NPU-owned HBM pages."""

    def prepare_planes(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        """Return separate Ascend NHD latent/RoPE or K/V planes without copying."""
        planes: list[torch.Tensor] = []
        for value in kv_caches.values():
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError(
                    "xDS Ascend requires separate block-major K and V planes"
                )
            planes.extend(value)
        if not planes or any(t.ndim != 4 for t in planes):
            raise ValueError("xDS Ascend requires NHD KV planes")
        return tuple(planes)

    def configure_device(self, device: torch.device) -> None:
        """Select the physical Ascend device used by HAL page registration."""
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        physical = int(visible.split(",")[device.index]) if visible else device.index
        os.environ["XDS_ASCEND_DEVICE_ID"] = str(physical)

    def create_pool(self, tensors: tuple[torch.Tensor, ...]) -> RegisteredNPUKVPool:
        """Register the owned padded Ascend allocations once per worker."""
        return RegisteredNPUKVPool(self._client, tensors)

    def synchronize_device(self) -> None:
        """Fence NPU users of the final pages before DMA and lease release."""
        torch.npu.synchronize(self._device)
