#!/usr/bin/env python3
import sys
import os
import multiprocessing
from unittest.mock import MagicMock

# Force 'fork' on macOS to match Linux behavior and avoid pickling errors
# Note: 'fork' is considered unsafe on macOS but is required for this legacy codebase structure.
try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

# Force eventlet to use 'poll' instead of 'kqueue' on Mac to avoid multiprocessing crashes
os.environ['EVENTLET_HUB'] = 'poll'

# 1. Mock Hardware Modules BEFORE importing main
sys.modules['picamera2'] = MagicMock()

# Mock serial with proper structure to avoid "MagicMock > int" errors
mock_serial = MagicMock()
mock_serial_instance = MagicMock()
mock_serial_instance.in_waiting = 0  # Crucial: Must be an int, not a MagicMock
mock_serial_instance.is_open = True
mock_serial.Serial.return_value = mock_serial_instance
sys.modules['serial'] = mock_serial

sys.modules['serial.tools'] = MagicMock()
sys.modules['serial.tools.list_ports'] = MagicMock()
sys.modules['rospy'] = MagicMock()

# Mock IpManager to avoid "hostname -I" error on Mac
sys.modules['src.dashboard.components.ip_manger'] = MagicMock()

# Mock psutil to avoid cpu_affinity error on Mac (not supported on macOS)
# And to provide dummy data for sensors which don't exist on Mac
mock_psutil = MagicMock()
mock_process = MagicMock()
# Create a valid return value for cpu_affinity (e.g. [0, 1, 2, 3])
mock_process.cpu_affinity = MagicMock(return_value=list(range(4)))
mock_psutil.Process.return_value = mock_process
mock_psutil.cpu_count.return_value = 4
# Mock virtual_memory().percent
mock_mem = MagicMock()
mock_mem.percent = 50.0
mock_psutil.virtual_memory.return_value = mock_mem
# Mock sensors_temperatures()['cpu_thermal'][0].current
# This structure is deep, so we need to construct it
mock_temp_entry = MagicMock()
mock_temp_entry.current = 45.0
mock_psutil.sensors_temperatures.return_value = {'cpu_thermal': [mock_temp_entry]}

sys.modules['psutil'] = mock_psutil

# 2. Set Environment
os.environ['RAVEN_SIMULATION'] = 'false'

# 3. Import and Run Main
try:
    print("🚀 Starting RAVEN Brain on Mac (Mocked Hardware)...")
    import main
    # main.py runs on import if it lacks a main() function wrapper
except ImportError as e:
    print(f"❌ Startup Failed (ImportError): {e}")
    print("\n💡 TIP: It looks like some dependencies are missing.")
    print("   If you are running on Mac, make sure to use the virtual environment:")
    print("   $ ./venv/bin/python run_mac.py")
    print("\n   Or install the Mac requirements first:")
    print("   $ pip install -r requirements-mac.txt")
except Exception as e:
    print(f"❌ Startup Failed (Exception): {e}")
