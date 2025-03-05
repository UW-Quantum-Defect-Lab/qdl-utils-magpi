# This is originally copied from https://github.com/qt3uw/qt3RFSynthControl/blob/main/src/qt3rfsynthcontrol/device.py
# Putting here to remove dependency on pip install qt3rfsynthcontrol
import os
import sys
import collections
import windfreak
import logging

logger = logging.getLogger(__name__)

# def discover_devices():
#     '''
#     Returns a list of discovered devices.

#     Each row in the list contains the
#         port, device description, hardware id.

#     Find your device and use the port value to instantiate a Pulser object.

#     '''

#     import serial.tools.list_ports
#     if os.name == 'nt':  # sys.platform == 'win32':
#         from serial.tools.list_ports_windows import comports
#     elif os.name == 'posix':
#         from serial.tools.list_ports_posix import comports

#     iterator = sorted(comports(include_links=True))
#     devices = [[port, desc, hwid] for port, desc, hwid in iterator]
#     return devices


class windfreak_synth_hd:

    def __init__(self, port):
        self._port = port
        self._inst = None
        self.last_write_command = None
        self.open()
        self._command_history = collections.deque(maxlen=1000)
        self.logger = logging.getLogger(__name__ + '.windfreak_synth_hd')

## PRIVATE
    def __del__(self):
        self.close()

    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc_value, traceback ):
        self.close()

    def __repr__(self):
        return f"windfreak_synth_hd({self._port})"

    def __str__(self):
        return print(self.hw_info())


    def _readlines(self):
        '''
        Read from device.

        Returns:
            a list of strings from the device.
        '''
        rdata = self._inst._dev.readlines()
        return [x.decode('utf-8').strip() for x in rdata]

## PUBLIC

    def open(self):
        if self._inst is None:
            self._inst = windfreak.SynthHD(self._port)

    def close(self):
        if self._inst is not None:
            self._inst.close()
            self._inst = None

    def hw_info(self):
        queries = ['model_type','serial_number','fw_version','hw_version','sub_version']
        return [(x,self._inst.read(x)) for x in queries]

    def current_status(self):
        self._inst._write('?')
        status = self._readlines()
        return status

    def set_continuous_sweep(self, continuous = 1):
        self._inst.write('sweep_cont',continuous)

    def start_sweep(self):
        self._inst.write('sweep_single',1)

    def stop_sweep(self):
        self._inst.write('sweep_single',0)

    def trigger_mode(self, mode='disabled'):
        assert mode in ['disabled', 'enabled']
        self._inst.trigger_mode = mode

    def set_power(self, channel, power):
        self._inst[channel].power = power

    def set_frequency(self, channel, frequency):
        self._inst[channel].frequency = frequency

    def set_channel_fixed_output(self, channel, power = None, frequency = None):
        '''
        Sets the power, frequency for a particular channel (0 = A, 1 = B)

        This turns OFF any frequency sweep that is currently running and disables
        external triggering.

        WARNING: this function is to be deprecated. Call each function separately:
        stop_sweep()
        trigger_mode('disabled')
        set_power(power)
        set_frequency(frequency)
        '''
        logger.warning('set_channel_fixed_output is to be deprecated. Call `stop_sweep()`, `trigger_mode("disabled")`, `set_power(p)` and set_frequency(f), separately as needed.')

        if power:
            self._inst[channel].power = power
        if frequency:
            self._inst[channel].frequency = frequency

    def rf_on(self, channel):
        '''
        Turns RF on channel.

        Powers the PLL, the amplifiers and unmutes the RF output.

        This uses the windfreak-python's `enable` function, which executes all three commands,
        equivalent to 'C{channel}E1r1h1'

        According to documentation, takes about 20 ms for power up.
        '''
        self._inst[channel].enable = True

    def rf_off(self, channel):
        '''
        Turns RF on channel.

        Powers off the PLL, off the amplifiers and mutes the RF output.

        This uses the windfreak-python's `enable` function, which executes all three commands,
        equivalent to 'C{channel}E0r0h0'

        According to documentation, takes about 20 ms for power down.
        '''
        self._inst[channel].enable = False

    def set_frequency_sweep(self, channel, power, frequency_low,
                            frequency_high, n_steps, trigger_mode = 'disabled',
                            trigger_polarity = 'high', frequency_sample_time = 0.500):
        '''
        The units of power is in dBm.
        The units of frequency_low and frequency_high are in Hz.
        The unit of frequency_sample_time is in seconds.

        The power is set to a constant value for all frequencies.

        The channel power is turned off during configuration. You can immediately
        enable the RF power by setting `enable = True` when calling this method.
        Otherwise, call `rf_on(channel)` after calling this method.

        Freqeuncy sweep is inclusive of frequency_low and frequency_high values.
        If n_steps = 2, then the RF will output at two distinct frequencies:
        frequency_low and frequency_high. Only values of n_steps >= 2 are allowed.

        If trigger_mode = 'disabled', then the RF generator will be set to
        visit each frequency for a duration of `frequency_sample_time`, which
        is set to 0.500 seconds, by default.

        If trigger_mode = 'single frequency step', then the RF generator will
        step to the next frequency each time a trigger pulse is received.
        You are responsible for setting up the external trigger device
        and submitting trigger pulses.
        IMPORTANT: The width of your trigger pulse should be shorter than
        frequency_sample_time, otherwise the RF generator will step to the next
        frequency.

        If trigger_mode = 'full frequency sweep', then the RF generator start
        a full sweep when a trigger pulse is received.
        You are responsible for setting up the external trigger device
        and submitting trigger pulses.
        If your trigger pulse is still ON when the sweep completes, the RF
        generator will start another sweep.

        This function only supports 'disabled', 'single frequency step',
        or 'full frequency sweep' for now.

        '''

        assert frequency_low < frequency_high
        assert n_steps >= 2
        assert trigger_mode in ['disabled', 'single frequency step', 'full frequency sweep']
        assert frequency_sample_time > 0.004
        assert frequency_sample_time < 10
        #ensures we are communicating with appropriate channel.
        self._inst.write('channel',channel)

        #turn off any sweeps & disable external trigger
        self._inst.write('sweep_single',0)
        self._inst.write('sweep_cont',0)
        self._inst.trigger_mode = 'disabled'

        frequency_step = (frequency_high - frequency_low) / (n_steps - 1)
        if trigger_polarity == 'high':
            self._inst._write('Y1') ## SETS the trigger polarity Y1 to HIGH, Y0 to LOW. This function is undocumented!
        else:
            self._inst._write('Y0')
        self._inst[channel].write('sweep_type',0) #X0 #linear sweep
        self._inst[channel].write('sweep_cont',0) #c0 #1 - set to do continuous sweep, 0 -- set to stop after full sweep
        #mw_source[0].trigger_mode ='disabled' #w0
        self._inst.trigger_mode = trigger_mode #w2

        self._inst[channel].write('sweep_direction',1) # force sweep from low to high

        self._inst[channel].write('sweep_freq_low', frequency_low / 1e6) #in MHz. Might need to be careful here on how many digits are written. for example, the device might not be happy with 10.000000000000000001
        self._inst[channel].write('sweep_freq_high', frequency_high / 1e6) #in MHz. Might need to be careful here on how many digits are written. for example, the device might not be happy with 10.000000000000000001
        self._inst[channel].write('sweep_freq_step', frequency_step / 1e6) #in MHz. Might need to be careful here on how many digits are written. for example, the device might not be happy with 10.000000000000000001

        # set power
        self._inst[channel].write('power', power) #f'W{scan_power:2.3f}'
        self._inst[channel].write('sweep_power_low',power) #f'[{scan_power:2.3f}'
        self._inst[channel].write('sweep_power_high',power) #f']{scan_power:2.3f}'

        #freq_step_time = 1 / data_rate #in seconds
        self._inst[channel].write('sweep_time_step',frequency_sample_time * 1e3) #limits between 4ms and 10000ms!

        total_time_in_s = frequency_sample_time * (n_steps)

        self.logger.info(f'Configured scan with')
        self.logger.info( f'scan_power = {self._inst[channel].power}')
        self.logger.info(f'scan_frequencies (low to high) = {self._inst[channel].read("sweep_freq_low")*1e-6} - {self._inst[channel].read("sweep_freq_high")*1e-6} MHz')
        self.logger.info(f'frequency step size = {self._inst[channel].read("sweep_freq_step")*1e-6} MHz')
        self.logger.info(f'time @ each frequency = {self._inst[channel].read("sweep_time_step")*1e-3} s')
        self.logger.info(f'number of steps, including low and high = {n_steps}')
        if trigger_mode in ['disabled', 'full frequency sweep']:
            self.logger.info(f'time for full scan = {total_time_in_s} s')
        self.logger.info(f'trigger mode = {self._inst.trigger_mode}')

    def run_sweep(self, channel, on=True):
        '''
        Starts the frequency sweep.

        Use this if `trigger_mode == disabled` to tell the RF Synth to begin the
        frequency sweep.

        If on = False, this stops a currently running sweep.

        This is equivalent to the (`sweep_single`, on) command in windfreak-python,
        which is equivalent to directly writing `g1`/`g0` to the serial port.

        '''
        self._inst[channel].write('sweep_single',1 if on else 0)

    @property
    def synthHD(self):
        return self._inst