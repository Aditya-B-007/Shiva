import os
import socket
import logging
from typing import Any, Dict
from .base import PerceptionDevice
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.network")

# Optional dependency check for system network metrics
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

try:
    from jnius import autoclass
    android_available = True
except ImportError:
    android_available = False

try:
    from rubicon.objc import ObjCClass
    ios_available = True
except ImportError:
    ios_available = False


class NetworkManager(PerceptionDevice):
    """
    SOLID-compliant Network & Security perception device.
    Monitors connectivity, bluetooth status, network interface traffic,
    and checks for hacking attempts / security anomalies.
    
    Exposes ONLY the 'capture()' interface for external execution.
    """

    @property
    def name(self) -> str:
        return "NetworkManager"

    @property
    def description(self) -> str:
        return "Captures system network connectivity, bluetooth status, traffic statistics, and security threats."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {}

    def capture(self, **kwargs: Any) -> Dict[str, Any]:
        """
        The ONLY externally exposed method to extract network and security metrics.
        Delegates to internal private sub-methods maintaining Single Responsibility.
        """
        current_os = get_current_os()
        logger.info(f"Gathering network and security statistics on: {current_os}")

        return {
            "internet_connected": self._check_internet_connectivity(),
            "mobile_data_connected": self._check_mobile_data(current_os),
            "bluetooth_enabled": self._check_bluetooth(current_os),
            "traffic_statistics": self._gather_packet_traffic(),
            "security_alerts": self._run_security_audit(current_os)
        }

    # =========================================================================
    # Internal Private Helper Routines (Maintaining Single Responsibility)
    # =========================================================================

    def _check_internet_connectivity(self) -> bool:
        """Checks if the system has active internet access using DNS resolution."""
        try:
            # Resolve cloudflare primary DNS host
            socket.setdefaulttimeout(2.0)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("1.1.1.1", 53))
            return True
        except Exception:
            return False

    def _check_mobile_data(self, current_os: str) -> bool:
        """Determines if mobile data connection is active."""
        if current_os == "android" and android_available:
            try:
                Context = autoclass("android.content.Context")
                activity = autoclass("org.kivy.android.PythonActivity").mActivity
                connectivity_manager = activity.getSystemService(Context.CONNECTIVITY_SERVICE)
                active_network = connectivity_manager.getActiveNetworkInfo()
                if active_network is not None:
                    # Type 0 is ConnectivityManager.TYPE_MOBILE
                    return active_network.getType() == 0 and active_network.isConnected()
            except Exception as e:
                logger.warning(f"Error querying Android mobile data: {e}")
        elif current_os == "ios" and ios_available:
            try:
                CTTelephonyNetworkInfo = ObjCClass("CTTelephonyNetworkInfo")
                telephony_info = CTTelephonyNetworkInfo.alloc().init()
                service_status = telephony_info.serviceCurrentRadioAccessTechnology
                return service_status is not None and len(service_status) > 0
            except Exception as e:
                logger.warning(f"Error querying iOS mobile data: {e}")
        return False

    def _check_bluetooth(self, current_os: str) -> bool:
        """Checks if Bluetooth hardware is active."""
        if current_os == "android" and android_available:
            try:
                BluetoothAdapter = autoclass("android.bluetooth.BluetoothAdapter")
                adapter = BluetoothAdapter.getDefaultAdapter()
                if adapter is not None:
                    return adapter.isEnabled()
            except Exception as e:
                logger.warning(f"Error querying Android bluetooth: {e}")
        elif current_os == "ios" and ios_available:
            try:
                CBCentralManager = ObjCClass("CBCentralManager")
                # Conforming state: CBCentralManagerStatePoweredOn corresponds to 5
                manager = CBCentralManager.alloc().initWithDelegate_queue_(None, None)
                return manager.state == 5
            except Exception as e:
                logger.warning(f"Error querying iOS bluetooth: {e}")
        return False

    def _gather_packet_traffic(self) -> Dict[str, Any]:
        """Collects dynamic network packet transmission counters."""
        if psutil_available:
            try:
                io_counters = psutil.net_io_counters()
                return {
                    "bytes_sent": io_counters.bytes_sent,
                    "bytes_received": io_counters.bytes_recv,
                    "packets_sent": io_counters.packets_sent,
                    "packets_received": io_counters.packets_recv,
                }
            except Exception as e:
                logger.warning(f"Error reading psutil packet counters: {e}")
        return {"bytes_sent": 0, "bytes_received": 0, "packets_sent": 0, "packets_received": 0}

    def _run_security_audit(self, current_os: str) -> Dict[str, Any]:
        """
        Scans for potential security threats, root/jailbreak indicators,
        suspicious listening ports, and unauthorized proxy routing.
        """
        alerts = []
        
        # 1. Root / Jailbreak Detection Checks
        is_rooted = False
        if current_os == "android":
            # Standard directories containing SU binaries on Android
            su_paths = [
                "/system/app/Superuser.apk", "/sbin/su", "/system/bin/su",
                "/system/xbin/su", "/data/local/xbin/su", "/data/local/bin/su",
                "/system/sd/xbin/su", "/system/bin/failsafe/su", "/data/local/su"
            ]
            is_rooted = any(os.path.exists(p) for p in su_paths)
        elif current_os == "ios":
            # Common iOS jailbreak file paths
            jailbreak_paths = [
                "/Applications/Cydia.app", "/Library/MobileSubstrate/MobileSubstrate.dylib",
                "/bin/bash", "/usr/sbin/sshd", "/etc/apt", "/private/var/lib/apt/"
            ]
            is_rooted = any(os.path.exists(p) for p in jailbreak_paths)
        
        if is_rooted:
            alerts.append(f"Device appears to be rooted or jailbroken ({current_os}).")

        # 2. Proxy and VPN checks
        vpn_indicators = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
        active_proxies = [var for var in vpn_indicators if var in os.environ]
        if active_proxies:
            alerts.append(f"Suspicious proxy variables active in environment: {active_proxies}")

        # 3. Port Scanning & Suspicious Local Port Activity
        open_ports_count = 0
        if psutil_available:
            try:
                # Count current active TCP connections listening for incoming traffic
                connections = psutil.net_connections(kind="inet")
                listening_connections = [conn for conn in connections if conn.status == "LISTEN"]
                open_ports_count = len(listening_connections)
                
                # Flag anomalies if abnormal counts of open ports are detected
                if open_ports_count > 25:
                    alerts.append(f"High volume of open listening ports ({open_ports_count}). Potential local scanning threat.")
            except Exception as e:
                logger.warning(f"Error querying active ports: {e}")

        return {
            "threats_detected": len(alerts) > 0,
            "alerts": alerts,
            "open_listening_ports": open_ports_count,
            "environment_secure": not is_rooted and len(alerts) == 0
        }
