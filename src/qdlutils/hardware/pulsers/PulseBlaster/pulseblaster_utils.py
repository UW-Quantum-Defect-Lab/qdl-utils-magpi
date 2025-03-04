

try:
    import qdlutils.hardware.pulsers.PulseBlaster.spinapi as pb_spinapi
    # import pulseblaster.spinapi as pb_spinapi
except NameError as e:
    print('spinapi did not load. Message: ' + str(e))
    pb_spinapi = None

from qdlutils.errors import PulseBlasterInitError, PulseBlasterError


class pb_controls():

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


