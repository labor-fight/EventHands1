"""CPU-only ownership/query failure tests; no NVIDIA library is required."""
from __future__ import annotations

import copy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from tools.gpu_borrow import monitor as M


UUID = "GPU-2812b716-420c-46b8-7812-fbe48f2c7682"
UUID2 = "GPU-39a910a1-7942-a849-4d85-d5007ec25de6"
XML = f"""<nvidia_smi_log>
  <attached_gpus>1</attached_gpus>
  <gpu><uuid>{UUID}</uuid><minor_number>99</minor_number>
    <mig_mode><current_mig>Disabled</current_mig></mig_mode>
    <fb_memory_usage><used>14 MiB</used></fb_memory_usage>
    <utilization><gpu_util>0 %</gpu_util></utilization>
    <processes>
      <process_info><pid>123</pid><type>C</type>
        <process_name>python</process_name><used_memory>0 MiB</used_memory>
      </process_info>
      <process_info><pid>456</pid><type>G</type>
        <process_name>/usr/lib/xorg/Xorg</process_name><used_memory>4 MiB</used_memory>
      </process_info>
    </processes>
  </gpu>
</nvidia_smi_log>"""


class FakeNvmlError(Exception):
    def __init__(self, message="", value=None):
        super().__init__(message)
        self.value = value


class FakeNotSupported(FakeNvmlError):
    pass


def fake_nvml() -> SimpleNamespace:
    """The driver-facing mock deliberately returns overlapping C/G contexts."""
    return SimpleNamespace(
        NVMLError=FakeNvmlError,
        NVMLError_NotSupported=FakeNotSupported,
        NVML_ERROR_NOT_SUPPORTED=3,
        NVML_ERROR_FUNCTION_NOT_FOUND=13,
        NVML_ERROR_ARGUMENT_VERSION_MISMATCH=25,
        nvmlMemory_v2=2,
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetCount=Mock(return_value=1),
        nvmlDeviceGetHandleByIndex=Mock(return_value="device-handle"),
        nvmlDeviceGetUUID=Mock(return_value=UUID.encode()),
        nvmlDeviceGetMigMode=Mock(return_value=(0, 0)),
        nvmlDeviceGetMemoryInfo=Mock(return_value=SimpleNamespace(used=14 * 1048576)),
        nvmlDeviceGetUtilizationRates=Mock(return_value=SimpleNamespace(gpu=0)),
        nvmlDeviceGetComputeRunningProcesses=Mock(
            return_value=[SimpleNamespace(pid=123, usedGpuMemory=0)]),
        nvmlDeviceGetGraphicsRunningProcesses=Mock(
            return_value=[SimpleNamespace(pid=123), SimpleNamespace(pid=456)]),
    )


class XmlTests(unittest.TestCase):
    def test_zero_memory_compute_context_and_graphics_are_both_preserved(self):
        gpu = M.parse_xml(XML, lambda pid: {123: 1007, 456: 1004}[pid])[0]
        self.assertEqual(gpu.uuid, UUID)
        self.assertEqual(gpu.index, 0)  # The Linux device minor is 99.
        self.assertEqual(gpu.processes, (M.Process(123, 1007, "C"), M.Process(456, 1004, "G")))
        self.assertEqual(gpu.used_mib, 14)

    def test_unselected_gpu_process_ownership_is_not_inspected(self):
        root = ET.fromstring(XML)
        second = copy.deepcopy(root.find("gpu"))
        second.find("uuid").text = UUID2
        second.find("processes/process_info/pid").text = "999"
        root.append(second)
        root.find("attached_gpus").text = "2"
        text = ET.tostring(root, encoding="unicode")

        def owner(pid):
            if pid == 999:
                raise PermissionError("unrelated GPU process is hidden")
            return 1007

        self.assertEqual(list(M.parse_xml(text, owner, uuids=(UUID,))), [0])
        with self.assertRaises(M.MonitorError):
            M.parse_xml(text, owner)
        # Selecting GPU 1 preserves its physical index instead of renumbering.
        self.assertEqual(list(M.parse_xml(text, lambda _pid: 1007, uuids=(UUID2,))), [1])

    def test_selected_gpu_missing_or_empty_selection_is_unknown(self):
        for selected in ((UUID2,), ()):
            with self.subTest(selected=selected), self.assertRaises(M.MonitorError):
                M.parse_xml(XML, lambda _pid: 1007, uuids=selected)

    def test_vanished_or_unreadable_process_is_unknown_not_idle(self):
        for error in (FileNotFoundError("exited"), PermissionError("hidden"), ValueError("UID")):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(M.MonitorError):
                    M.parse_xml(XML, Mock(side_effect=error))

    def test_unknown_owner_uid_is_rejected(self):
        for uid in (-1, None, "1007"):
            with self.subTest(uid=uid), self.assertRaises(M.MonitorError):
                M.parse_xml(XML, lambda _pid: uid)

    def test_missing_or_unavailable_fields_cannot_look_idle(self):
        broken = [
            XML.replace("<processes>", "<unknown>").replace("</processes>", "</unknown>"),
            XML.replace("<pid>123</pid>", ""),
            XML.replace("<type>C</type>", ""),
            XML.replace("<gpu_util>0 %</gpu_util>", ""),
            XML.replace("<used>14 MiB</used>", "<used>N/A</used>"),
            XML.replace("<gpu_util>0 %</gpu_util>", "<gpu_util>N/A</gpu_util>"),
            XML.replace("<gpu_util>0 %</gpu_util>", "<gpu_util>101 %</gpu_util>"),
            XML.replace("<pid>123</pid>", "<pid>0</pid>"),
            XML.replace("<type>C</type>", "<type>unknown</type>"),
            XML.replace("<attached_gpus>1</attached_gpus>", "<attached_gpus>2</attached_gpus>"),
            XML.replace(UUID, "GPU-invalid"),
            "<nvidia_smi_log>",
        ]
        for number, text in enumerate(broken):
            with self.subTest(case=number), self.assertRaises(M.MonitorError):
                M.parse_xml(text, lambda _pid: 1007)

    def test_enabled_mig_and_mps_cannot_hide_process_ownership(self):
        for text in (
            XML.replace("Disabled", "Enabled"),
            XML.replace("<process_name>python</process_name>",
                        "<process_name>/usr/bin/nvidia-cuda-mps-server</process_name>"),
        ):
            with self.subTest(xml=text), self.assertRaises(M.MonitorError):
                M.parse_xml(text, lambda _pid: 1007)

    def test_actual_uid_reader_checks_effective_uid_and_mps(self):
        for status in (
            "Name:\tpython\nUid:\t1007\t0\t0\t0\n",
            "Name:\tpython\n",
            "Name:\tnvidia-cuda-mps\nUid:\t1007\t1007\t1007\t1007\n",
        ):
            with patch.object(M.Path, "read_text", return_value=status):
                with self.subTest(status=status), self.assertRaises(M.MonitorError):
                    M.read_uid(123)
        with patch.object(M.Path, "read_text", return_value="Uid:\t1007\t1007\t1007\t1007\n"):
            self.assertEqual(M.read_uid(123), 1007)


class NvmlTests(unittest.TestCase):
    def test_unselected_gpu_stats_and_processes_are_not_queried(self):
        nvml = fake_nvml()
        nvml.nvmlDeviceGetCount.return_value = 2
        nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda index: index
        nvml.nvmlDeviceGetUUID.side_effect = lambda handle: (UUID, UUID2)[handle]

        def only_gpu_zero(handle):
            if handle != 0:
                raise FakeNvmlError("unrelated GPU query failed")
            return [SimpleNamespace(pid=123)]

        nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = only_gpu_zero
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M, "read_uid", return_value=1007):
            monitor = M.NvidiaMonitor("nvml")
            self.assertEqual(list(monitor.snapshot((UUID,))), [0])
            nvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(0, version=nvml.nvmlMemory_v2)
            nvml.nvmlDeviceGetUtilizationRates.assert_called_once_with(0)
            nvml.nvmlDeviceGetGraphicsRunningProcesses.assert_called_once_with(0)
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()
            monitor.close()

    def test_v2_driver_reserved_memory_does_not_count_as_used(self):
        nvml = fake_nvml()

        def memory_info(_handle, version=None):
            if version == nvml.nvmlMemory_v2:
                return SimpleNamespace(used=14 * 1048576, reserved=611 * 1048576)
            return SimpleNamespace(used=625 * 1048576)

        nvml.nvmlDeviceGetMemoryInfo.side_effect = memory_info
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M, "read_uid", return_value=1007):
            monitor = M.NvidiaMonitor("nvml")
            self.assertEqual(monitor.snapshot()[0].used_mib, 14)
            nvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(
                "device-handle", version=nvml.nvmlMemory_v2,
            )
            monitor.close()

    def test_unsupported_v2_falls_back_to_conservative_v1_value(self):
        nvml = fake_nvml()

        def memory_info(_handle, version=None):
            if version is not None:
                raise FakeNvmlError("v2 unsupported", value=nvml.NVML_ERROR_NOT_SUPPORTED)
            return SimpleNamespace(used=625 * 1048576)

        nvml.nvmlDeviceGetMemoryInfo.side_effect = memory_info
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M, "read_uid", return_value=1007):
            monitor = M.NvidiaMonitor("nvml")
            # Do not subtract a guessed reserve on drivers without the v2 API.
            self.assertEqual(monitor.snapshot()[0].used_mib, 625)
            self.assertEqual(nvml.nvmlDeviceGetMemoryInfo.call_count, 2)
            monitor.close()

    def test_v2_driver_failure_does_not_fall_back(self):
        nvml = fake_nvml()
        nvml.nvmlDeviceGetMemoryInfo.side_effect = FakeNvmlError("GPU lost", value=15)
        with patch.dict(sys.modules, {"pynvml": nvml}):
            monitor = M.NvidiaMonitor("nvml")
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()
            self.assertEqual(nvml.nvmlDeviceGetMemoryInfo.call_count, 1)
            monitor.close()

    def test_nvml_missing_selected_gpu_is_unknown(self):
        nvml = fake_nvml()
        with patch.dict(sys.modules, {"pynvml": nvml}):
            monitor = M.NvidiaMonitor("nvml")
            with self.assertRaises(M.MonitorError):
                monitor.snapshot((UUID2,))
            nvml.nvmlDeviceGetComputeRunningProcesses.assert_not_called()
            monitor.close()

    def test_merges_compute_and_graphics_by_pid_and_closes_once(self):
        nvml = fake_nvml()
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M, "read_uid", return_value=1007):
            monitor = M.NvidiaMonitor()
            self.assertEqual(monitor.backend, "nvml")
            gpu = monitor.snapshot()[0]
            self.assertEqual(gpu.processes, (M.Process(123, 1007, "C+G"), M.Process(456, 1007, "G")))
            self.assertEqual(gpu.uuid, UUID)
            monitor.close()
            monitor.close()
            nvml.nvmlShutdown.assert_called_once()
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()

    def test_unsupported_mig_api_is_valid_for_non_mig_gpu(self):
        nvml = fake_nvml()
        nvml.nvmlDeviceGetMigMode.side_effect = FakeNotSupported()
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M, "read_uid", return_value=1007):
            monitor = M.NvidiaMonitor("nvml")
            self.assertEqual(len(monitor.snapshot()), 1)
            monitor.close()

    def test_enabled_mig_fails_closed(self):
        nvml = fake_nvml()
        nvml.nvmlDeviceGetMigMode.return_value = (1, 1)
        with patch.dict(sys.modules, {"pynvml": nvml}):
            monitor = M.NvidiaMonitor("nvml")
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()
            monitor.close()

    def test_driver_query_failure_never_falls_back_to_smi(self):
        nvml = fake_nvml()
        nvml.nvmlDeviceGetComputeRunningProcesses.side_effect = FakeNvmlError("driver failed")
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(M.subprocess, "run") as smi:
            monitor = M.NvidiaMonitor()
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()
            smi.assert_not_called()
            monitor.close()

    def test_vanished_process_is_unknown_in_nvml_too(self):
        nvml = fake_nvml()
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.object(
            M, "read_uid", side_effect=M.MonitorError("process exited"),
        ):
            monitor = M.NvidiaMonitor("nvml")
            with self.assertRaises(M.MonitorError):
                monitor.snapshot()
            monitor.close()

    def test_auto_only_falls_back_when_binding_is_missing(self):
        with patch.dict(sys.modules, {"pynvml": None}):
            self.assertEqual(M.NvidiaMonitor().backend, "smi")
            with self.assertRaises(M.MonitorError):
                M.NvidiaMonitor("nvml")
        nvml = fake_nvml()
        nvml.nvmlInit.side_effect = FakeNvmlError("initialization failed")
        with patch.dict(sys.modules, {"pynvml": nvml}), self.assertRaises(M.MonitorError):
            M.NvidiaMonitor()


class SmiTests(unittest.TestCase):
    def test_timeout_and_nonzero_exit_are_unknown(self):
        monitor = M.NvidiaMonitor("smi")
        for error in (subprocess.TimeoutExpired("nvidia-smi", 3),
                      subprocess.CalledProcessError(1, "nvidia-smi"), FileNotFoundError()):
            with self.subTest(error=type(error).__name__), patch.object(
                M.subprocess, "run", side_effect=error,
            ), self.assertRaises(M.MonitorError):
                monitor.snapshot()


if __name__ == "__main__":
    unittest.main()
