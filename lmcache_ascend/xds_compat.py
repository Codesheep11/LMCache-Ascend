# SPDX-License-Identifier: Apache-2.0
"""Pin xDS service buffers through AscendCL for the xDS LMCache fork."""

import ctypes

import torch
from lmcache.v1.memory_management import (
    TensorMemoryAllocator,
    XDSServiceMemoryAllocator,
)


class AscendHostRegistration:
    def __init__(self):
        self.library = ctypes.CDLL("libascendcl.so")
        self.library.aclrtHostRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.aclrtHostRegister.restype = ctypes.c_int
        self.library.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
        self.library.aclrtHostUnregister.restype = ctypes.c_int

    def cudaHostUnregister(self, pointer):
        # The inherited allocator uses this narrow host-registration interface.
        return self.library.aclrtHostUnregister(pointer)


class AscendXDSServiceMemoryAllocator(XDSServiceMemoryAllocator):
    def __init__(self, client, size, align_bytes, close_client=False):
        self.client = client
        self.size = size
        self.align_bytes = align_bytes
        self.buffer_ref = None
        self._view = None
        self._cudart = AscendHostRegistration()
        self._host_registered = False
        self._close_client = close_client
        self._client_closed = False
        self._closed = False
        torch.npu.set_device(torch.npu.current_device())
        view, buffer_ref = client.alloc_buffer(size)
        self._view = view
        self.buffer_ref = int(buffer_ref)
        tensor = torch.frombuffer(view, dtype=torch.uint8).flatten()
        self.base_ptr = tensor.data_ptr()
        mapped = ctypes.c_void_p()
        status = self._cudart.library.aclrtHostRegister(
            self.base_ptr, size, 0, ctypes.byref(mapped)
        )
        if status:
            tensor = None
            view.release()
            client.free_buffer(self.buffer_ref)
            self.buffer_ref = None
            raise RuntimeError(f"AscendCL xDS host registration failed: {status}")
        self._host_registered = True
        TensorMemoryAllocator.__init__(self, tensor, align_bytes=align_bytes)
