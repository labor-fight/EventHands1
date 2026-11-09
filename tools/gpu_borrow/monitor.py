"""Read-only NVIDIA snapshots using NVML, with an nvidia-smi XML fallback.

Indices follow the order of ``gpu`` elements in the full nvidia-smi XML output,
not Linux device minor numbers. Callers must retain the full GPU UUID when
selecting devices. Graphics contexts are included (``Process.kind == 'G'``): a
caller can distinguish the machine's small Xorg baseline from compute work.

An incomplete query, vanished/unreadable PID, unsupported MIG/MPS context, or
unknown field raises MonitorError. None of those states means that a GPU is
idle. Snapshots observe existing contexts; they cannot reserve GPU memory or
prevent another process from allocating between two observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Callable
import xml.etree.ElementTree as ET


class MonitorError(RuntimeError):
    """The query cannot safely establish device/process ownership."""


@dataclass(frozen=True)
class Process:
    pid: int
    uid: int
    kind: str = "C"


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    used_mib: int
    utilization: int
    processes: tuple[Process, ...]


def read_uid(pid: int) -> int:
    """Return a process UID, treating process-exit races as an unknown sample."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        if any(line.startswith("Name:") and "nvidia-cuda-mps" in line
               for line in status.splitlines()):
            raise ValueError("MPS client ownership is unsupported")
        fields = next(line.split()[1:] for line in status.splitlines()
                      if line.startswith("Uid:"))
        if len(fields) != 4:
            raise ValueError("malformed Uid field")
        real_uid, effective_uid, _, _ = map(int, fields)
        if real_uid != effective_uid:
            raise ValueError("different real/effective UIDs")
        return real_uid
    except (OSError, StopIteration, ValueError) as exc:
        raise MonitorError(f"Cannot determine owner of GPU process {pid}: {exc}") from exc


def _text(element: ET.Element, path: str) -> str:
    value = element.findtext(path)
    if value is None or not value.strip():
        raise MonitorError(f"Missing NVIDIA XML field: {path}")
    return value.strip()


def _quantity(value: str, unit: str) -> int:
    match = re.fullmatch(r"(\d+)\s*" + re.escape(unit), value)
    if not match:
        raise MonitorError(f"Unknown NVIDIA quantity: {value!r}")
    return int(match.group(1))


def parse_xml(
    text: str, uid_reader: Callable[[int], int] = read_uid,
    uuids: tuple[str, ...] | None = None,
) -> dict[int, GPU]:
    """Parse selected devices; unrelated process ownership is never inspected.

    Selection uses complete UUIDs, while returned indices retain their position
    in the entire GPU inventory. Injectable UID lookup supports CPU tests.
    """
    selected = None if uuids is None else set(uuids)
    if selected is not None and not selected:
        raise MonitorError("GPU selection is empty")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise MonitorError(f"Invalid NVIDIA XML: {exc}") from exc
    if root.tag != "nvidia_smi_log":
        raise MonitorError("Unexpected NVIDIA XML root")
    elements = root.findall("gpu")
    try:
        attached = int(_text(root, "attached_gpus"))
    except ValueError as exc:
        raise MonitorError("Invalid attached_gpus count") from exc
    if attached < 1 or attached != len(elements):
        raise MonitorError("Incomplete or empty NVIDIA GPU list")

    result: dict[int, GPU] = {}
    seen_uuids: set[str] = set()
    owners: dict[int, int] = {}
    for index, element in enumerate(elements):
        uuid = _text(element, "uuid")
        if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
            raise MonitorError(f"Invalid GPU UUID: {uuid!r}")
        if uuid in seen_uuids:
            raise MonitorError(f"Duplicate GPU UUID: {uuid}")
        seen_uuids.add(uuid)
        if selected is not None and uuid not in selected:
            continue
        mig = element.findtext("mig_mode/current_mig")
        if mig is not None and mig.strip() not in {"Disabled", "N/A"}:
            raise MonitorError(f"GPU {index}: MIG contexts are unsupported")
        used_mib = _quantity(_text(element, "fb_memory_usage/used"), "MiB")
        utilization = _quantity(_text(element, "utilization/gpu_util"), "%")
        if utilization > 100:
            raise MonitorError(f"GPU {index}: invalid utilization")
        process_list = element.find("processes")
        if process_list is None:
            raise MonitorError(f"GPU {index}: missing process list")
        if process_list.text and process_list.text.strip() not in {"None", ""}:
            raise MonitorError(f"GPU {index}: unavailable process list")
        processes: list[Process] = []
        for process in process_list.findall("process_info"):
            try:
                pid = int(_text(process, "pid"))
            except ValueError as exc:
                raise MonitorError(f"GPU {index}: invalid process PID") from exc
            if pid <= 0:
                raise MonitorError(f"GPU {index}: invalid process PID {pid}")
            kind = _text(process, "type")
            if kind not in {"C", "G", "C+G"}:
                raise MonitorError(f"GPU {index}: unsupported process type {kind!r}")
            name = _text(process, "process_name")
            if "nvidia-cuda-mps" in name:
                raise MonitorError(f"GPU {index}: MPS client ownership is unsupported")
            if pid not in owners:
                try:
                    owners[pid] = uid_reader(pid)
                except MonitorError:
                    raise
                except (OSError, ValueError) as exc:
                    raise MonitorError(f"Cannot determine owner of GPU process {pid}: {exc}") from exc
                if not isinstance(owners[pid], int) or owners[pid] < 0:
                    raise MonitorError(f"GPU process {pid}: invalid owner UID")
            processes.append(Process(pid, owners[pid], kind))
        result[index] = GPU(index, uuid, used_mib, utilization, tuple(processes))
    if selected is not None and selected - seen_uuids:
        raise MonitorError(f"Selected GPUs are missing: {sorted(selected - seen_uuids)}")
    return result


class NvidiaMonitor:
    """Prefer the official nvidia-ml-py binding; XML needs only stdlib.

    ``auto`` falls back only when the optional Python binding is not installed.
    A driver/query failure is never silently treated as an empty device list.
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend not in {"auto", "nvml", "smi"}:
            raise ValueError(f"Unknown NVIDIA monitoring backend: {backend}")
        self._nvml = None
        if backend != "smi":
            try:
                import pynvml
            except ImportError as exc:
                if backend == "nvml":
                    raise MonitorError("NVML binding missing; install nvidia-ml-py") from exc
            else:
                try:
                    pynvml.nvmlInit()
                except pynvml.NVMLError as exc:
                    raise MonitorError(f"NVML initialization failed: {exc}") from exc
                self._nvml = pynvml
        self.backend = "nvml" if self._nvml is not None else "smi"

    def _snapshot_nvml(self, uuids: tuple[str, ...] | None = None) -> dict[int, GPU]:
        nvml = self._nvml
        assert nvml is not None
        selected = None if uuids is None else set(uuids)
        if selected is not None and not selected:
            raise MonitorError("GPU selection is empty")
        result: dict[int, GPU] = {}
        owners: dict[int, int] = {}
        seen_uuids: set[str] = set()
        try:
            count = nvml.nvmlDeviceGetCount()
            if count < 1:
                raise MonitorError("Empty NVML GPU list")
            for index in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                uuid = nvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("ascii")
                if not isinstance(uuid, str) or not re.fullmatch(
                    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid,
                ):
                    raise MonitorError(f"GPU {index}: invalid NVML UUID")
                if uuid in seen_uuids:
                    raise MonitorError(f"Duplicate NVML UUID: {uuid}")
                seen_uuids.add(uuid)
                if selected is not None and uuid not in selected:
                    continue
                try:
                    current_mig, _ = nvml.nvmlDeviceGetMigMode(handle)
                except nvml.NVMLError_NotSupported:
                    current_mig = 0
                if current_mig:
                    raise MonitorError(f"GPU {index}: MIG contexts are unsupported")
                # v1 includes driver-reserved memory in `used` on this driver;
                # v2 separates `reserved` and agrees with nvidia-smi's used FB
                # memory. Keeping the v1 value on older drivers is conservative
                # (it can prevent borrowing, but cannot invent free memory).
                memory_version = getattr(nvml, "nvmlMemory_v2", None)
                memory = None
                if memory_version is not None:
                    try:
                        memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=memory_version)
                    except TypeError:
                        # Older Python bindings do not accept the version kwarg.
                        pass
                    except nvml.NVMLError as exc:
                        unsupported = {
                            getattr(nvml, name, -1) for name in (
                                "NVML_ERROR_NOT_SUPPORTED", "NVML_ERROR_FUNCTION_NOT_FOUND",
                                "NVML_ERROR_ARGUMENT_VERSION_MISMATCH",
                            )
                        }
                        if getattr(exc, "value", None) not in unsupported:
                            raise
                if memory is None:
                    memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                if memory.used < 0 or not 0 <= utilization <= 100:
                    raise MonitorError(f"GPU {index}: invalid NVML memory/utilization")
                kinds: dict[int, set[str]] = {}
                for kind, query in (
                    ("C", nvml.nvmlDeviceGetComputeRunningProcesses),
                    ("G", nvml.nvmlDeviceGetGraphicsRunningProcesses),
                ):
                    for process in query(handle):
                        pid = int(process.pid)
                        if pid <= 0:
                            raise MonitorError(f"GPU {index}: invalid NVML process PID")
                        kinds.setdefault(pid, set()).add(kind)
                processes = []
                for pid, types in sorted(kinds.items()):
                    if pid not in owners:
                        owners[pid] = read_uid(pid)
                    processes.append(Process(pid, owners[pid], "+".join(sorted(types))))
                result[index] = GPU(
                    index, uuid, (int(memory.used) + 1048575) // 1048576,
                    utilization, tuple(processes),
                )
            if selected is not None and selected - seen_uuids:
                raise MonitorError(f"Selected GPUs are missing: {sorted(selected - seen_uuids)}")
        except MonitorError:
            raise
        except (nvml.NVMLError, OSError, ValueError, TypeError, UnicodeError) as exc:
            raise MonitorError(f"NVML query failed: {exc}") from exc
        return result

    def snapshot(self, uuids: tuple[str, ...] | None = None) -> dict[int, GPU]:
        if self.backend == "nvml":
            if self._nvml is None:
                raise MonitorError("NVML monitor is closed")
            return self._snapshot_nvml(uuids)
        try:
            response = subprocess.run(
                ["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
                timeout=3, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MonitorError(f"NVIDIA query failed: {exc}") from exc
        return parse_xml(response.stdout, uuids=uuids)

    def close(self) -> None:
        if self._nvml is not None:
            nvml, self._nvml = self._nvml, None
            try:
                nvml.nvmlShutdown()
            except nvml.NVMLError as exc:
                raise MonitorError(f"NVML shutdown failed: {exc}") from exc
