try:
    import qdlutils.hardware.pulsers.PulseBlaster.spinapi as pb_spinapi
except NameError as e:
    print('spinapi did not load. Message: ' + str(e))
    pb_spinapi = None
import numpy as np

class PBInd:
    """
    PBInd is a client of the SpinCore PulseBlaster API that allows the user
    to program the pins of the PulseBlaster (PB) independently of one another in python.
    Features validation for sequences based on the provided PulseBlaster specifications.    
    
    :param pins: List of pin numbers to be programmed on the PB (0-23 are valid)
    :param DEBUG_MODE: Flag to enable debug mode (prints out PB instructions)
    :param on_time: Cycle duration (ns)
    :param resolution: Granularity of pulse increments (see your PB specs)
    :param minimum_pulse: Minimum pulse width (see your PB specs)
    :param auto_stop: Flag to automatically stop the PulseBlaster after a sequence (1 for enabled, 0 for disabled).
    :raises ValueError: Cycle duration (on_time) is not a multiple of resolution.
    """
    def __init__(self,
                pins,
                DEBUG_MODE = 0,
                on_time = 4000,
                resolution = 10,
                minimum_pulse = 50,
                auto_stop = 0):
        self._pins = pins
        self._DEBUG_MODE = DEBUG_MODE
        if (on_time%10 != 0):
            raise ValueError('total instruction time is not a multiple of 10ns: ' + str(on_time) + 'ns')
        else:
            self._on_time = on_time
        self._res  = resolution
        self._minimum_pulse = minimum_pulse
        self._smps = round(self._on_time/self._res) # total length of instruction in samples (before looping)
        self._output_chs = ['0'*self._smps]*(len(self._pins)) # 'bitset' representation of output
        self.instructions = "" # string representation of commands issued to Pulseblaster
        self._auto_stop = auto_stop # if turned off, allows the client to program after a call to PBInd.program()
        self.spinapi = spincore_spinapi

    def on(self, pin, start, length):
        """
        Sets a selected pin to high voltage.
        
        :param pin (int): Selected pulseblaster pin
        :param start (int): Start time (ns) relative to the start of the cycle
        :param length (int): Duration (ns)
        :raises ValueError: Start or length are not multiples of the pulse resolution.
        :raises IndexError: Programmed start time falls outside of the valid cycle range.
        """
        self._set(pin, start, length, 1)

    def off(self, pin, start, length):
        """
        Sets a selected pin to low voltage.
        
        :param pin (int): Selected pulseblaster pin
        :param start (int): Start time (ns) relative to the start of the cycle
        :param length (int): Duration (ns)
        :raises ValueError: Start or length are not multiples of the pulse resolution.
        :raises IndexError: Programmed start time falls outside of the valid cycle range.
        """
        self._set(pin, start, length, 0)

    def make_clock(self, pin, period):
        """
        Configures a selected pin to output a 50% duty-cycle clock with the given period.
        
        :param pin (int): Pin to be used as the clock output.
        :param period (int): Clock period duration (ns).
        :raises ValueError: If the clock period is shorter than twice the minimum pulse length, 
             or not a multiple of twice the resolution.
        """
        # Ensure period is long enough
        if period / 2 < self._minimum_pulse:
            raise ValueError(f'Requested clock period ({period} ns) is too short: less than {2 * self._minimum_pulse} ns')
        
        # Ensure period is a multiple of resolution*2
        if period % (self._res * 2) != 0:
            raise ValueError(f'Requested clock period ({period} ns) is not a multiple of ({self._res * 2} ns)')

        ticks = 0  # count the number of actual ticks that are programmed
        cursor = 0
        # Sweep across chs and add ticks
        while cursor + period <= self._on_time:
            self.on(pin, cursor, period / 2)
            cursor = cursor + period / 2
            self.off(pin, cursor, period / 2)
            cursor = cursor + period / 2
            ticks = ticks + 1


    def program(self, loops):
        """
        Summary of the function or class.
        
        :param loops: Number of cycle repititions
        :returns: Description of the return value.
        :raises ExceptionType: Description of the exception raised.
        """
        if loops < 1:
            raise Exception('number of loops must be positive integer')

        if not self._DEBUG_MODE & self._auto_stop:
            self.spinapi.pb_start_programming('PULSE_PROGRAM')

        # the client did not request delays, so the matrix is unchanged
        self._validate_pulse_instructions(self._output_chs)
        self._write_instruction(self._output_chs, loops)

        if self._DEBUG_MODE:
            self.instructions = self.instructions + "pb_inst_pbonly(0, 'STOP', 0, " + str(2*self._res) + ")\n"

        if (not self._DEBUG_MODE) & self._auto_stop:
            self.spinapi.pb_inst_pbonly(0, 'STOP', 0, 2 * self._res)
            self.spinapi.pb_stop_programming()

        self.print_instructions()

    def print_instructions(self):
        """
        Prints out the string representation of the instructions issued to the Pulseblaster if _DEBUG_MODE is enabled
        """
        if self._DEBUG_MODE:
            print(self.instructions)

    ## PRIVATE
    def _write_instruction(self, actual_chs, loops):
        """Produces the instructions used to program the PulseBlaster from the CHS matrix"""
        if len(actual_chs)==0:
            return
        # Given a chs matrix, program a sequence of Pulseblaster instructions. 
        # The 'command' can be self.spinapi.Inst.CONTINUE or self.spinapi.Inst.LOOP
        prev_state_d = 0  # index of 'prev_state'
        prev_state =  self._get_state(prev_state_d, actual_chs, len(self._pins)) # vertical slice of states as bitset

        actual_smps = len(actual_chs[0])

        cur_command = self.spinapi.Inst.CONTINUE  # the next instruction
        last_command = self.spinapi.Inst.CONTINUE
        if loops == float('inf'):
            # in this case we want an infinite loop
            last_command = self.spinapi.Inst.BRANCH
            loops=0
        elif loops > 1:
            # if loops > 1, then the first and last commands will be loop commands
            cur_command = self.spinapi.Inst.LOOP
            last_command = self.spinapi.Inst.END_LOOP

        first_inst = float('inf')  # this will eventually change to the first instruction ID

        for d in range(1,actual_smps):
            current_state = self._get_state(d, actual_chs, len(self._pins))
            if self._DEBUG_MODE:
                print(current_state)
            compare = current_state == prev_state
            if not compare:
                # at least one channel changed state, issue new instruction to PB
                hex_flag = self._hex_flag(prev_state)
                duration = (d - prev_state_d) * self._res
                if not self._DEBUG_MODE:
                    first_inst = min(self.spinapi.pb_inst_pbonly(hex_flag, cur_command, int(loops), duration * self.spinapi.ns), first_inst)  # we want inst to be the lowest instruction ID
                else:
                    first_inst = 0

                if self._DEBUG_MODE:
                    self.instructions = self.instructions + "pb_inst_pbonly(" + str(hex_flag) + "," + str(cur_command) + "," + str(loops) + "," + str(duration * self.spinapi.ns) + ")\n"

                cur_command = self.spinapi.Inst.CONTINUE  # even if loops > 1, the next commands will all be CONTINUE commands (until last END_LOOP command)
                prev_state = current_state
                prev_state_d = d

        # we have reached the end of the chs matrix. Now issue the last instruction
        hex_flag = self._hex_flag(prev_state)
        duration = (actual_smps - prev_state_d) * self._res  # the plus one is needed otherwise there is an off by one error
        if (first_inst == float('inf')) & (loops > 1):
            # in this case, the matrix was homogeneous and no
            # instructions were issued in the loop. Therefore the last
            # command CANNOT be an END_LOOP (there was no 'begin loop').
            # Simply change it to a CONTINUE command
            last_command = self.spinapi.Inst.CONTINUE
            duration = duration * loops

        if not self._DEBUG_MODE:
            if first_inst == float('inf'):
                first_inst = 0
            self.spinapi.pb_inst_pbonly(hex_flag, last_command, first_inst, duration * self.spinapi.ns)  # this instruction could be an END_LOOP command

        if self._DEBUG_MODE:
            self.instructions = self.instructions + "pb_inst_pbonly(" + str(hex_flag)+ "," + str(last_command) +","+ str(first_inst) +","+str(duration) +")\n"

    def _set(self, pin, start, len, val):
        """
        Sets a selected pin to a specifified value (high/low).
        
        :param pin (int): Selected pulseblaster pin
        :param start (int): Start time (ns) relative to the start of the cycle
        :param length (int): Duration (ns)
        :raises ValueError: Start or length are not multiples of the pulse resolution.
        :raises IndexError: Programmed start time falls outside of the valid cycle range.
        """
        if start % self._res != 0:
            raise ValueError(f"Start time is not a multiple of 10ns: {start}ns")
        if len % self._res != 0:
            raise ValueError(f"Length is not a multiple of 10ns: {len}ns")
        start_smp = round(start / self._res)
        stop_smp = start_smp + round(len / self._res)-1

        # Check for valid sample times
        if start_smp < 0 or start_smp > self._smps:
            raise IndexError(f"Invalid start sample time: {start_smp}")
        elif stop_smp > self._smps:
            raise IndexError(f"Invalid stop sample time: {stop_smp}")

        if stop_smp >= start_smp:
            ch = self._get_ch(pin)
            self._output_chs[ch] = self._output_chs[ch][min(start_smp,0):start_smp] + str(val)*(stop_smp-start_smp+1) + self._output_chs[ch][stop_smp:-1]
            # TODO: possibly add rounding feature to improve downsample

    def _get_ch(self, pin):
        index = -1
        for d in range(len(self._pins)):
            if self._pins[d] == pin:
                index = d
                break

        if index == -1:
            raise Exception('invalid pin requested: ' + str(pin))

        return index

    def _hex_flag(self, state):
        hex_flag = 0
        for d in range(len(state)):
            if state[d] == '1':
                # this pin is on, add to flag
                # TODO: somewhere, clarify self._pins(d) vs 2^self._pins(d)
                hex_flag = hex_flag|2**(self._pins[d])
        return hex_flag

    def _get_state(self, state_index, state_array, num_elements):
            stateslice=""
            for i in range(num_elements):
                stateslice = stateslice + state_array[i][state_index]

            return stateslice

    def _validate_pulse_instructions(self, chs_matrix):
        """
        Validates that all instructions respect PulseBlaster timing constraints
        
        :param chs_matrix: 2D matrix representing pulse instructions, where each row corresponds to a channel and each column
                        corresponds to a time step. The values in the matrix indicate the state (high or low) of the channels
                        at each time step
        :raises Exception: Matrix value changes more than once in a rolling period equal to the minimum pulse length
        """
        min_instruction_span = int(self._minimum_pulse / self._res)  # Required minimum span in columns

        chs_matrix = np.array([list(row) for row in chs_matrix])
    
        current_streak = 1
        streak_start = 0
        change_indices = []
        min_streak = float('inf')
        min_streak_index = 0
        problem_channels_start = []  # Will store which channels started the shortest instruction
        problem_channels_end = []  # Will store which channels ended the shortest instruction
    
        # Compare entire columns
        for i in range(1, chs_matrix.shape[1]):
            if np.array_equal(chs_matrix[:, i], chs_matrix[:, i-1]):
                current_streak += 1
            else:
                # Column changed, check if previous streak was long enough
                if current_streak < min_streak:
                    min_streak = current_streak
                    min_streak_index = streak_start
                    # Find which channels changing at this change interface and prior interface
                    if (streak_start > 0):
                        problem_channels_start = np.where(chs_matrix[:, streak_start] != chs_matrix[:, streak_start-1])[0]
                    else:
                        problem_channels_start = np.array([]) # All channels change at start
                    problem_channels_end = np.where(chs_matrix[:, i] != chs_matrix[:, i-1])[0]
            
                change_indices.append(i)
                streak_start = i
                current_streak = 1
    
        # Check final streak
        if current_streak < min_streak:
            min_streak = current_streak
            min_streak_index = streak_start
            if streak_start > 0:
                problem_channels_start = np.where(chs_matrix[:, streak_start] != chs_matrix[:, streak_start-1])[0]
            else:
                problem_channels_start = np.array([])
            problem_channels_end = np.array([])
    
        # If minimum streak is too short, raise exception
        if min_streak < min_instruction_span:
            start_channels_str = "initial state" if len(problem_channels_start) == 0 else problem_channels_start.tolist()
            end_channels_str = "final state" if len(problem_channels_end) == 0 else problem_channels_end.tolist()
        
            error_msg = (f'Instruction duration {min_streak * self._res}ns shorter than required '
                        f'{self._minimum_pulse}ns starting at time {min_streak_index * self._res}ns (index {min_streak_index}). '
                        f'Instructions changed at indices: {change_indices}. '
                        f'Channels that changed at instruction\'s start: {start_channels_str}. '
                        f'Channels that changed at instruction\'s end: {end_channels_str}.')
            
            raise Exception(error_msg)
