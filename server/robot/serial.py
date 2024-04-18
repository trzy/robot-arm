#
# serial.py
# Bart Trzynadlowski
#
# Identification of usable serial ports.
#

import glob
import sys
from typing import List

import serial


def serial_ports() -> List[str]:
    """ 
    Lists serial port names.

    Returns
    -------
    List[str]
        List of serial port names.
    
    Raises
    ------
    EnvironmentError
        Unsupported or unknown platform.
    """
    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # Excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def find_serial_port(port_pattern: str) -> str | None:
    """
    Finds a serial port by name.

    Parameters
    ----------
    port_pattern : str
        Serial port search pattern (supports wildcards: *, ?, [sequence], [!sequence]).
    
    Returns
    -------
    str | None
        Serial port name or None if no port found.
    """
    # If port contains wildcard characters, we need to compare against all ports, otherwise we just
    # return the port the user specified
    if "*" in port_pattern or "?" in port_pattern or "[" in port_pattern:
        import fnmatch
        ports = serial_ports()
        matches = [port for port in ports if fnmatch.fnmatch(name=port, pat=port_pattern)]
        if len(matches) == 0:
            print("Error: No matching ports found")
            return None
        if len(matches) > 1:
            print(f"Error: Multiple ports match given pattern: {', '.join(ports)}")
            return None
        return matches[0]
    else:
        return port_pattern