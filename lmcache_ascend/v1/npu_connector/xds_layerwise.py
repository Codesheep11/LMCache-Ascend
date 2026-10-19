# SPDX-License-Identifier: Apache-2.0
"""KV-major layerwise persistence and final-block xDS DMA on Ascend 910B."""

# Standard
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any, Optional

# Third Party
import torch
from lmcache.v1.gpu_connector.xds_readv import (
    XDSReadvPlan,
    plan_layerwise_readv_batch,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

# First Party
from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
    XDSNPURegistrationMixin,
    get_npu_readv_registration,
    get_xds_kv_tensors,
    is_310p,
)
import lmcache_ascend.c_ops as lmc_ops


def _layerwise_format(caches: list[Any]) -> tuple[KVCacheFormat, bool]:
    """Recognize vLLM NHD layers, including lists normalized by the base class."""
    first = caches[0]
    if (
        isinstance(first, (tuple, list))
        and len(first) == 2
        and all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in first)
        and first[0].shape == first[1].shape
    ):
        return KVCacheFormat.SEPARATE_KV, False
    if isinstance(first, torch.Tensor) and first.ndim == 5:
        if first.shape[0] == 2:
            return KVCacheFormat.MERGED_KV, True
        if first.shape[1] == 2:
            return KVCacheFormat.MERGED_KV, False
    raise ValueError("xDS layerwise requires separate or merged NHD KV")


class XDSReadvLayerwiseNPUConnector(
    XDSNPURegistrationMixin, VLLMPagedMemLayerwiseNPUConnector
):
    """Use explicit KV_2TD storage with readv into owned final KV blocks.

    Selected only for opt-in xDS layerwise readv. Unsupported spans return None
    before DMA; the fallback respects each object's explicit storage format.
    """

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the persisted [K/V, token, hidden] shape for num_tokens."""
        return torch.Size([2, num_tokens, self.hidden_dim_size])

    def plan_layerwise_xds_readv(
        self,
        chunks: Sequence[
            tuple[list[torch.Size], list[torch.dtype], MemoryFormat, int, int]
        ],
        io_unit: int,
        **kwargs: Any,
    ) -> Optional[list[list[XDSReadvPlan]]]:
        """Validate every layer/chunk and return fresh direct destination plans.

        Return None for MLA, 310P, unsupported layouts, partial blocks or invalid
        registration ownership. The caller retains block ownership until drain.
        """
        if is_310p() or self.use_mla:
            return None
        self.initialize_kvcaches_ptr(**kwargs)
        caches = self.kvcaches
        slots = kwargs.get("slot_mapping_cpu", kwargs.get("slot_mapping"))
        if (
            not caches
            or len(caches) != self.num_layers
            or not isinstance(slots, torch.Tensor)
            or any(kwargs.get("vllm_cached_tokens", 0) > c[3] for c in chunks)
        ):
            return None
        pool = self.get_xds_readv_pool(caches)
        if pool is not None:
            tensors = pool.tensors
            layout = pool.layout
            ranges = pool.registration_ranges
        else:
            try:
                tensors, layout = get_xds_kv_tensors(caches)
                ranges = tuple(get_npu_readv_registration(t) for t in tensors)
            except ValueError:
                return None
        plans = plan_layerwise_readv_batch(
            tensors, chunks, slots, io_unit, layout, "npu", ranges
        )
        if plans is not None and pool is not None:
            for layer, layer_plans in enumerate(plans):
                indices = (
                    (2 * layer, 2 * layer + 1) if layout == "SEPARATE" else (layer,)
                )
                refs = tuple(pool.buffer_refs[i] for i in indices)
                plans[layer] = [replace(plan, buffer_refs=refs) for plan in layer_plans]
        return plans

    def prepare_xds_readv_dma(self, device: torch.device) -> None:
        """Finish previous NPU users before overwriting the caller-owned blocks."""
        if device.type != "npu":
            raise ValueError("Ascend readv requires NPU target memory")
        torch.npu.synchronize(device)

    def finish_xds_readv_dma(self, device: torch.device) -> None:
        """Synchronize the coherent 910B path after storage completion."""
        if device.type != "npu":
            raise ValueError("Ascend readv requires NPU target memory")
        torch.npu.synchronize(device)

    def batched_from_gpu(
        self,
        memory_objs: list[list[MemoryObj]],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> Iterator[None]:
        """Gather each layer to KV-major staging objects with the Ascend kernel.

        Yield num_layers+1 times, preserving the layerwise storage protocol.
        Missing KV tensors or unsupported formats raise ValueError before use.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if not self.kvcaches:
            raise ValueError("kvcaches must be supplied before layerwise storage")
        fmt, two_major = _layerwise_format(self.kvcaches)
        caches = (
            [tuple(layer) for layer in self.kvcaches]
            if fmt == KVCacheFormat.SEPARATE_KV
            else self.kvcaches
        )
        slots = kwargs["slot_mapping"]
        current = torch.npu.current_stream()
        for layer, objects in enumerate(memory_objs):
            with torch.npu.stream(self.store_stream):
                self.store_stream.wait_stream(current)
                for obj, start, end in zip(objects, starts, ends, strict=True):
                    if obj.tensor is None or obj.tensor.shape != self.get_shape(
                        end - start
                    ):
                        raise ValueError("xDS layerwise KV-major store shape mismatch")
                    obj.metadata.fmt = MemoryFormat.KV_2TD
                    lmc_ops.single_layer_kv_transfer(
                        obj.tensor,
                        caches[layer],
                        slots[start:end],
                        True,
                        fmt.value,
                        False,
                        two_major,
                    )
            yield
            if kwargs.get("sync", False):
                self.store_stream.synchronize()
        yield

    def batched_to_gpu(
        self, starts: list[int], ends: list[int], **kwargs: Any
    ) -> Iterator[None]:
        """Load unsupported direct plans using the explicit object format.

        Accept KV_2TD and legacy KV_T2D, rejecting any other format. The generator
        consumes one list of objects per layer and preserves stream ordering.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if not self.kvcaches:
            raise ValueError("kvcaches must be supplied before layerwise loading")
        fmt, two_major = _layerwise_format(self.kvcaches)
        caches = (
            [tuple(layer) for layer in self.kvcaches]
            if fmt == KVCacheFormat.SEPARATE_KV
            else self.kvcaches
        )
        slots = kwargs["slot_mapping"]
        current = torch.npu.current_stream()
        for layer in range(self.num_layers):
            objects = yield
            if kwargs.get("sync", False):
                current.wait_stream(self.load_stream)
            with torch.npu.stream(self.load_stream):
                for obj, start, end in zip(objects, starts, ends, strict=True):
                    if obj.metadata.fmt not in (
                        MemoryFormat.KV_2TD,
                        MemoryFormat.KV_T2D,
                    ):
                        raise ValueError("unsupported xDS layerwise source format")
                    lmc_ops.single_layer_kv_transfer(
                        obj.tensor,
                        caches[layer],
                        slots[start:end],
                        False,
                        fmt.value,
                        obj.metadata.fmt == MemoryFormat.KV_T2D,
                        two_major,
                    )
                events = kwargs.get("h2d_copy_completion_events")
                if events is not None:
                    event = torch.npu.Event()
                    event.record(self.load_stream)
                    events.append(event)
        yield
        if kwargs.get("sync", False):
            current.wait_stream(self.load_stream)
        yield
