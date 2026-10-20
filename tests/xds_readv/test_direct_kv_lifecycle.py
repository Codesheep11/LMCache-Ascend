# SPDX-License-Identifier: Apache-2.0
"""Direct stores must remain inside the serving worker's synchronous hook."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import lmcache_ascend  # noqa: F401
import pytest
import torch
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


@pytest.mark.parametrize("asynchronous", [False, True])
def test_store_uses_direct_backend_before_ascend_gather(asynchronous):
    engine = object.__new__(AscendLMCacheEngine)
    engine.config = LMCacheEngineConfig.from_defaults()
    engine.config.xds_backend = True
    engine.is_store_async = asynchronous
    with patch.object(LMCacheEngine, "store") as direct_store:
        if asynchronous:
            with pytest.raises(ValueError, match="wait_for_save"):
                engine.store(tokens=[1, 2])
            direct_store.assert_not_called()
        else:
            engine.store(tokens=[1, 2])
            direct_store.assert_called_once_with([1, 2], None, None, None)


def test_direct_dma_error_is_not_swallowed_by_save_hook():
    from lmcache_ascend.integration.vllm.vllm_v1_adapter import (
        LMCacheAscendConnectorV1Impl,
        LMCacheConnectorMetadata,
    )

    engine = MagicMock()
    engine.store.side_effect = RuntimeError("unresolved DMA")
    adapter = object.__new__(LMCacheAscendConnectorV1Impl)
    adapter.config = LMCacheEngineConfig.from_defaults()
    adapter.config.xds_backend = True
    adapter.kv_role = "kv_both"
    adapter.use_layerwise = False
    adapter.kv_caches = {"layer": torch.ones(1)}
    adapter._manager = SimpleNamespace(lmcache_engine=engine)
    adapter._lmcache_chunk_size = 128
    request = SimpleNamespace(
        req_id="r",
        token_ids=list(range(128)),
        slot_mapping=torch.arange(128),
        save_spec=SimpleNamespace(can_save=True, skip_leading_tokens=0),
        is_last_prefill=True,
        disagg_spec=None,
        request_configs={},
    )
    adapter._parent = MagicMock()
    adapter._parent._get_connector_metadata.return_value = LMCacheConnectorMetadata(
        [request]
    )
    with patch("torch.npu.Event"):
        with pytest.raises(RuntimeError, match="unresolved DMA"):
            adapter.wait_for_save()


@pytest.mark.parametrize("legacy_staging", [False, True])
def test_connector_default_disables_payload_buffer(legacy_staging):
    from lmcache.utils import EngineType
    from lmcache_ascend.v1.npu_connector import CreateNPUConnector

    config = LMCacheEngineConfig.from_defaults()
    config.xds_backend = True
    if legacy_staging:
        config.extra_config = {"xds_direct_kv_io": False}
    metadata = SimpleNamespace(use_mla=False, worker_id=0)
    module = "lmcache_ascend.v1.npu_connector"
    with (
        patch(f"{module}.need_gpu_interm_buffer", return_value=True),
        patch("torch.npu.device_count", return_value=1),
        patch("torch.npu.set_device"),
        patch(f"{module}.VLLMPagedMemNPUConnectorV2.from_metadata") as factory,
    ):
        assert (
            CreateNPUConnector(config, metadata, EngineType.VLLM)
            is factory.return_value
        )
        assert factory.call_args.args[1] is legacy_staging
