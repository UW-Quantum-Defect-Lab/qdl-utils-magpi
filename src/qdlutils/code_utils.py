def discover_devices():
    '''
    Returns a list of discovered devices.

    Each row in the list contains the
        port, device description, hardware id.
        
    '''

    import serial.tools.list_ports
    import os
    if os.name == 'nt':  # sys.platform == 'win32':
        from serial.tools.list_ports_windows import comports
    elif os.name == 'posix':
        from serial.tools.list_ports_posix import comports

    iterator = sorted(comports(include_links=True))
    devices = [[port, desc, hwid] for port, desc, hwid in iterator]
    return devices

