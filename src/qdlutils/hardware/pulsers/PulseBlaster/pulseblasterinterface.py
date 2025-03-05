

try:
    import qdlutils.hardware.pulsers.PulseBlaster.spinapi as pb_spinapi
    # import pulseblaster.spinapi as pb_spinapi
except NameError as e:
    print('spinapi did not load. Message: ' + str(e))
    pb_spinapi = None

from qdlutils.errors import PulseBlasterInitError, PulseBlasterError


class PulseBlasterInterface():

    def start(self):
        self.open()
        ret = pb_spinapi.pb_start()
        if ret != 0:
            raise PulseBlasterError(f'{ret}: {pb_spinapi.pb_get_error()}')
        self.close()

    def stop(self):
        self.open()
        ret = pb_spinapi.pb_stop()
        if ret != 0:
            raise PulseBlasterError(f'{ret}: {pb_spinapi.pb_get_error()}')
        self.close()

    def reset(self):
        self.open()
        ret = pb_spinapi.pb_reset()
        if ret != 0:
            raise PulseBlasterError(f'{ret}: {pb_spinapi.pb_get_error()}')
        self.close()

    def close(self):
        ret = pb_spinapi.pb_close()
        if ret != 0:
            raise PulseBlasterError(f'{ret}: {pb_spinapi.pb_get_error()}')

    def stop_programming(self):
        if pb_spinapi.pb_stop_programming() != 0:
            raise PulseBlasterError(pb_spinapi.pb_get_error())

    def start_programming(self):
        if pb_spinapi.pb_start_programming(0) != 0:
            raise PulseBlasterError(pb_spinapi.pb_get_error())

    def open(self):
        pb_spinapi.pb_select_board(self.pb_board_number)
        ret = pb_spinapi.pb_init()
        print(f'pb_init returned {ret}')
        if ret != 0:
            self.close() #if opening fails, attempt to close before raising error
            raise PulseBlasterInitError(f'{ret}: {pb_spinapi.pb_get_error()}')
        pb_spinapi.pb_core_clock(100*pb_spinapi.MHz)



class PulseBlasterCWODMR(PulseBlasterInterface):
    '''
    Programs the pulse sequences needed for CWODMR.

    Provides an
      * always ON channel for an AOM.
      * 50% duty cycle pulse for RF switch
      * clock signal for use with a data acquisition card
      * trigger signal for use with a data acquisition card
    '''
    def __init__(self, pb_board_number = 1,
                       aom_channel = 0,
                       rf_channel = 1,
                       clock_channel = 2,
                       trigger_channel = 3,
                       rf_pulse_duration = 5e-6,
                       clock_period = 200e-9,
                       trigger_width = 500e-9):
        """
        pb_board_number - the board number (0, 1, ...)
        aom_channel output controls the AOM by holding a positive voltage
        rf_channel output controls a RF switch
        clock_channel output provides a clock input to the NI DAQ card
        trigger_channel output provides a rising edge trigger for the NI DAQ card
        """
        self.pb_board_number = pb_board_number
        self.aom_channel = aom_channel
        self.rf_channel = rf_channel
        self.clock_channel = clock_channel
        self.trigger_channel = trigger_channel
        self.rf_pulse_duration = np.round(rf_pulse_duration, 8)
        self.clock_period = np.round(clock_period, 8)
        self.trigger_width = np.round(trigger_width, 8)


    def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
        '''
        rf_pulse_duration is in seconds
        '''
        if rf_pulse_duration:
            self.raise_for_pulse_width(rf_pulse_duration)
            self.rf_pulse_duration = np.round(rf_pulse_duration,8)
        else:
            self.raise_for_pulse_width(self.rf_pulse_duration)

        cycle_length = 2*self.rf_pulse_duration

        hardware_pins = [self.aom_channel, self.rf_channel,
                         self.clock_channel, self.trigger_channel]

        self.open()
        pb = PBInd(pins = hardware_pins, on_time = int(cycle_length*1e9))
        self.start_programming()

        pb.on(self.trigger_channel, 0, int(self.trigger_width*1e9))
        pb.make_clock(self.clock_channel, int(self.clock_period*1e9))
        pb.on(self.aom_channel, 0, int(cycle_length*1e9))
        pb.on(self.rf_channel, 0, int(self.rf_pulse_duration*1e9))

        pb.program([],float('inf'))
        self.stop_programming()

        self.close()
        return np.round(cycle_length / self.clock_period).astype(int)


    def experimental_conditions(self):
        '''
        Returns a dictionary of paramters that are pertinent for the relevant experiment
        '''
        return {
            'rf_pulse_duration':self.rf_pulse_duration,
            'clock_period':self.clock_period
        }

    def raise_for_pulse_width(self, rf_pulse_duration, *args, **kwargs):
        if rf_pulse_duration < 50e-9:
            raise PulseTrainWidthError(f'RF width too small {int(rf_pulse_duration)} < 50 ns')

