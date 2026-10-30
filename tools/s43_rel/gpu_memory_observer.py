"""Read-only NVML device VRAM observations, including driver/context memory.

These periodic device readings are observed lower bounds on a short peak,
not an exact allocator peak or a process attribution on a shared device.
"""
import ctypes


class MemoryInfo(ctypes.Structure):
    _fields_ = [(key, ctypes.c_ulonglong) for key in ("total", "free", "used")]


class MemoryObserver:
    def __init__(self, physical_gpu):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)]
        self.library.nvmlShutdown.restype = ctypes.c_int
        self._check(self.library.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self._check(self.library.nvmlDeviceGetHandleByIndex_v2(int(physical_gpu), ctypes.byref(self.handle)))
        self.library.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        identity = ctypes.create_string_buffer(100)
        self._check(self.library.nvmlDeviceGetUUID(self.handle, identity, 100))
        self.uuid = identity.value.decode()
        self.physical_gpu = int(physical_gpu)
        self.count = 0
        self.peak = 0
        self.baseline = self.sample()

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError("NVML read-only memory query failed: " + str(code))

    def sample(self):
        value = MemoryInfo()
        self._check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(value)))
        self.count += 1
        self.peak = max(self.peak, int(value.used))
        self.last = int(value.used)
        return self.last

    def close(self):
        result = dict(physical_gpu=self.physical_gpu, gpu_uuid=self.uuid, samples=self.count,
            baseline_device_used_bytes=self.baseline, sampled_peak_device_used_bytes=self.peak,
            sampled_peak_minus_baseline_bytes=max(0, self.peak-self.baseline),
            final_device_used_bytes=self.last, sampling_period_seconds=.05,
            scope="NVML-v1 device used incl reserved memory and driver/context; device idle baseline is included. Increment above measured idle baseline is separate. Periodically observed peak, other occupants included, not exact process peak. Exact Torch allocator peak is separate.")
        self._check(self.library.nvmlShutdown())
        return result
