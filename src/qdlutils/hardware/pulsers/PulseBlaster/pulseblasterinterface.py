

try:
    import qdlutils.hardware.pulsers.PulseBlaster.spinapi as pb_spinapi
    # import pulseblaster.spinapi as pb_spinapi
except NameError as e:
    print('spinapi did not load. Message: ' + str(e))
    pb_spinapi = None

import numpy as np
import time
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
        # print(f'pb_init returned {ret}')
        if ret != 0:
            self.close() #if opening fails, attempt to close before raising error
            raise PulseBlasterInitError(f'{ret}: {pb_spinapi.pb_get_error()}')
        pb_spinapi.pb_core_clock(100*pb_spinapi.MHz)

    def raise_for_pulse_width(self, pulse_duration):
        if pulse_duration < 50e-9:
            raise ValueError('Pulse duration must be at least 50 ns')
        
    def run_the_pb_sequence(self):
        # The start() method wasn't working. This following sequence of commands 
        # was found to work. Why, tbd.
        pb_stop_programming_ret = pb_spinapi.pb_stop_programming()
        # print(f'pb_stop_programming_ret = {pb_stop_programming_ret}')
        pb_reset_ret = pb_spinapi.pb_reset()
        # print(f'pb_reset_ret = {pb_reset_ret}')
        pb_start_ret = pb_spinapi.pb_start()
        # print(f'pb_start_ret = {pb_start_ret}')
        pb_close_ret = pb_spinapi.pb_close()
        # print(f'pb_close_ret = {pb_close_ret}')
        pb_stop_ret = pb_spinapi.pb_stop()
        # print(f'pb_stop_ret = {pb_stop_ret}')

    def pb_all_off(self):
        self.reset()
        self.stop()
        self.open()
        self.start_programming()
        pb_spinapi.pb_inst_pbonly(
            0x0,
            pb_spinapi.Inst.BRANCH,
            0,
            1000
        )

        self.stop_programming()
        self.run_the_pb_sequence()
        time.sleep(0.1)
        self.reset()
        self.stop()

