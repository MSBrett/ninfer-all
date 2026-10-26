"""Host telemetry regressions; no GPU or inference server is required."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch


SOURCE = Path(__file__).resolve().parents[2] / "tools" / "monitor" / "monitor.py"
SPEC = importlib.util.spec_from_file_location("ninfer_monitor", SOURCE)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def make_monitor(windows=None):
    instance = monitor.Monitor.__new__(monitor.Monitor)
    instance._windows = windows
    instance.last_cpu_idle = None
    instance.last_cpu_total = None
    return instance


class FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


class WindowsBindingsTests(unittest.TestCase):
    def setUp(self):
        self.system = monitor.WindowsSystem.__new__(monitor.WindowsSystem)
        self.system.filetime = FileTime
        self.system.kernel32 = SimpleNamespace(GetSystemTimes=Mock(), GlobalMemoryStatusEx=Mock())

    def test_cpu_uses_64_bit_ticks_and_does_not_double_count_idle(self):
        values = (2**35 + 7, 2**36 + 13, 2**34 + 17)

        def fill_times(*pointers):
            for pointer, value in zip(pointers, values):
                pointer._obj.dwLowDateTime = value & 0xFFFFFFFF
                pointer._obj.dwHighDateTime = value >> 32
            return 1

        self.system.kernel32.GetSystemTimes.side_effect = fill_times
        self.assertEqual(self.system.cpu_times(), (values[0], values[1] + values[2]))

    def test_ram_uses_physical_bytes_and_binary_megabytes(self):
        total, available = 128 * 2**30 + 2**19, 20 * 2**30 + 2**18

        def fill_memory(pointer):
            status = pointer._obj
            # Native API requires the caller's ABI size before writing the result.
            self.assertEqual(status.dwLength, 64)
            status.ullTotalPhys = total
            status.ullAvailPhys = available
            status.ullTotalVirtual = 8 * 2**40
            status.ullAvailVirtual = 7 * 2**40
            status.ullTotalPageFile = 256 * 2**30
            status.ullAvailPageFile = 64 * 2**30
            return 1

        self.system.kernel32.GlobalMemoryStatusEx.side_effect = fill_memory
        self.assertEqual(self.system.ram(), {
            "used_mb": (total - available) / 2**20,
            "total_mb": total / 2**20,
        })

    def test_zero_used_ram_is_a_valid_sample(self):
        def fill_memory(pointer):
            pointer._obj.ullTotalPhys = 16 * 2**30
            pointer._obj.ullAvailPhys = 16 * 2**30
            return 1

        self.system.kernel32.GlobalMemoryStatusEx.side_effect = fill_memory
        self.assertEqual(self.system.ram(), {"used_mb": 0.0, "total_mb": 16384.0})

    def test_native_failures_raise_oserror(self):
        for method, api in ((self.system.cpu_times, self.system.kernel32.GetSystemTimes),
                            (self.system.ram, self.system.kernel32.GlobalMemoryStatusEx)):
            with self.subTest(method=method.__name__):
                api.return_value = 0
                with patch.object(ctypes, "get_last_error", return_value=5, create=True), \
                     patch.object(ctypes, "WinError", return_value=OSError(5, "denied"), create=True):
                    with self.assertRaises(OSError):
                        method()


class HostCpuTests(unittest.TestCase):
    def test_first_sample_then_interval_busy_fraction(self):
        # 300 total ticks elapsed; 225 were idle, therefore 25% were busy.
        windows = SimpleNamespace(cpu_times=Mock(side_effect=[(1000, 2000), (1225, 2300)]))
        instance = make_monitor(windows)
        self.assertIsNone(instance.read_cpu())
        self.assertAlmostEqual(instance.read_cpu(), 25.0)

    def test_idle_and_fully_busy_intervals_are_valid(self):
        windows = SimpleNamespace(cpu_times=Mock(side_effect=[(10, 20), (20, 30), (20, 40)]))
        instance = make_monitor(windows)
        self.assertIsNone(instance.read_cpu())
        self.assertEqual(instance.read_cpu(), 0.0)
        self.assertEqual(instance.read_cpu(), 100.0)

    def test_invalid_deltas_are_unavailable_then_rebaseline(self):
        for invalid in ((10, 20), (5, 15), (5, 30), (30, 30)):
            with self.subTest(invalid=invalid):
                idle, total = invalid
                windows = SimpleNamespace(cpu_times=Mock(side_effect=[
                    (10, 20), invalid, (idle + 5, total + 20)]))
                instance = make_monitor(windows)
                self.assertIsNone(instance.read_cpu())
                self.assertIsNone(instance.read_cpu())
                self.assertEqual(instance.read_cpu(), 75.0)

    def test_failed_read_does_not_replace_baseline(self):
        windows = SimpleNamespace(cpu_times=Mock(side_effect=[
            (10, 20), OSError("unavailable"), (20, 60)]))
        instance = make_monitor(windows)
        self.assertIsNone(instance.read_cpu())
        self.assertIsNone(instance.read_cpu())
        self.assertEqual(instance.read_cpu(), 75.0)

    def test_linux_iowait_is_idle_and_guest_is_not_counted_twice(self):
        instance = make_monitor()
        # First eight fields advance 100 ticks, of which idle+iowait advance 50.
        rows = ("cpu 100 10 40 200 30 5 5 10 50 5\n",
                "cpu 120 15 50 240 40 10 5 20 65 10\n")
        for index, row in enumerate(rows):
            with patch("builtins.open", mock_open(read_data=row)) as opened:
                result = instance.read_cpu()
                opened.assert_called_once_with("/proc/stat")
            if index == 0:
                self.assertIsNone(result)
            else:
                self.assertEqual(result, 50.0)

    def test_linux_missing_or_malformed_stat_is_unavailable(self):
        for error in (OSError("missing"), None):
            with self.subTest(error=error):
                reader = mock_open(read_data="cpu malformed\n")
                reader.side_effect = error
                with patch("builtins.open", reader):
                    self.assertIsNone(make_monitor().read_cpu())


class HostRamTests(unittest.TestCase):
    def test_windows_ram_is_returned_without_proc_access(self):
        value = {"used_mb": 0.0, "total_mb": 16384.0}
        instance = make_monitor(SimpleNamespace(ram=Mock(return_value=value)))
        with patch("builtins.open", side_effect=AssertionError("Windows must not read /proc")):
            self.assertEqual(instance.read_ram(), value)

    def test_windows_failure_is_unavailable(self):
        self.assertIsNone(make_monitor(SimpleNamespace(ram=Mock(side_effect=OSError()))).read_ram())

    def test_linux_ram_uses_available_instead_of_free(self):
        info = "MemTotal: 16777216 kB\nMemFree: 1048576 kB\nMemAvailable: 4194304 kB\n"
        with patch("builtins.open", mock_open(read_data=info)) as opened:
            self.assertEqual(make_monitor().read_ram(), {"used_mb": 12288.0, "total_mb": 16384.0})
            opened.assert_called_once_with("/proc/meminfo")

    def test_linux_failed_or_malformed_ram_is_unavailable(self):
        for error in (OSError("missing"), None):
            with self.subTest(error=error):
                reader = mock_open(read_data="MemTotal: malformed kB\n")
                reader.side_effect = error
                with patch("builtins.open", reader):
                    self.assertIsNone(make_monitor().read_ram())


@unittest.skipUnless(os.name == "nt", "native Windows API smoke check")
class NativeWindowsTests(unittest.TestCase):
    def test_real_windows_api_samples_are_sensible(self):
        instance = make_monitor(monitor.WindowsSystem())
        self.assertIsNone(instance.read_cpu())
        time.sleep(0.05)
        cpu = instance.read_cpu()
        self.assertIsNotNone(cpu)
        self.assertGreaterEqual(cpu, 0.0)
        self.assertLessEqual(cpu, 100.0)
        ram = instance.read_ram()
        self.assertIsNotNone(ram)
        self.assertGreater(ram["total_mb"], 0)
        self.assertGreaterEqual(ram["used_mb"], 0)
        self.assertLessEqual(ram["used_mb"], ram["total_mb"])


if __name__ == "__main__":
    unittest.main()
