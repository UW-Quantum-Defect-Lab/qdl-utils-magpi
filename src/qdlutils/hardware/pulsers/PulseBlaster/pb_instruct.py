import numpy as np

class PB_Channel():
    def __init__(
            self,
            pin = None,
            channel_name = None,
            delay = 0,
            pulse_extension = 0
            ):
        self.pin = pin
        self.channel_name = channel_name
        self.delay = delay
        self.pulse_extension = pulse_extension

    def __repr__(self):
        return f'PB_Channel(pin={self.pin}, channel_name={self.channel_name}, start_stop={self.start_stop}, delay={self.delay}, pulse_extension={self.pulse_extension})'

    def __str__(self):
        return f'PB_Channel(pin={self.pin}, channel_name={self.channel_name}, start_stop={self.start_stop}, delay={self.delay}, pulse_extension={self.pulse_extension})'

class PB_Instruct():
    def __init__(
            self,
            active_channels = None,
            cycle_period = None,
            clock_pin = None,
            clock_type = None,
            instruction_conflict_resolution_method = 'abort'
            ):
        '''
        active_channels: list of PB_Channel objects
        cycle_period: float, cycle period in seconds
        clock_pin: int, pin number of the clock channel
        clock_type: str, 'fixed_frequency' or 'arb_gate_free_falling_edge'
        instruction_conflict_resolution_method: str, 'abort', 'round_early', 'round_late', 'round_near'
        '''
        self.active_channels = active_channels
        self.cycle_period = cycle_period
        self.clock_pin = clock_pin
        self.clock_type = clock_type
        if clock_pin is not None and clock_type is None:
            raise ValueError('clock_type must be specified.')
        self.instruction_conflict_resolution_method = instruction_conflict_resolution_method

        self.num_active_channels = len(self.active_channels)

    def generate_instructions(self):
        num_channel_flips = 0

        for channel in self.active_channels:
            # Adjust start and stop times for delays and stretches
            try:
                channel.start_stop_actual = np.array([
                    [
                        np.round(start + channel.delay, 8),
                        np.round(stop + channel.delay + channel.pulse_extension, 8)
                    ] for start, stop in channel.start_stop
                ])
            except Exception as e:
                raise ValueError(f'Error in channel {channel.pin}: {e}. Check start_stop formatting!')

            # Check if any start or stop time is greater than self.cycle_period
            for start, stop in channel.start_stop_actual:
                if start > self.cycle_period or stop > self.cycle_period:
                    raise ValueError(f"Start or stop time {start, stop} is greater than self.cycle_period {self.cycle_period}. channel.pin: {channel.pin}")

            # Convert times to integer nanoseconds.
            channel.start_stop_actual_ns = np.round(channel.start_stop_actual * 1e9).astype(int)
            # Count the number of channel flips.
            num_channel_flips += 2 * len(channel.start_stop_actual)

        # Initialize the channel flip array. Elements are [pin, change (i.e., +/-1), start_ns].
        chflip_pin_change_startns = [[None] * 3] * num_channel_flips
        m_flip = 0
        for channel in self.active_channels:
            for pulse_start, pulse_stop in channel.start_stop_actual_ns:
                chflip_pin_change_startns[m_flip] = [channel.pin, 1, pulse_start]
                m_flip += 1
                chflip_pin_change_startns[m_flip] = [channel.pin, -1, pulse_stop]
                m_flip += 1

        # Sort the channel flips by start time.
        chflip_pin_change_startns.sort(key=lambda x: x[2])
        num_channel_flips = len(chflip_pin_change_startns)

        # %% Condition the channel flip times to satisfy the PulseBlaster constraints and eliminate conflicts
        # This is done by changing the start times of the pulses.

        # Clock constraint -- fixed frequency
        # Align non-clock channel start times with clock channel start times if within 60 ns
        if self.clock_pin is not None and self.clock_type == 'fixed_frequency':
            for m_flip in range(num_channel_flips):
                # Get the info of the flip event that might need to be changed
                pin, change, start_ns = chflip_pin_change_startns[m_flip]
                # It will only be changed if it is not the clock pin.
                if pin != self.clock_pin:
                    for clock_chflip in chflip_pin_change_startns:
                        # Get the info of the flip that might be a clock channel flip.
                        clock_pin, clock_change, clock_start_ns = clock_chflip
                        # Check if it is actually a clock channel flip and if the start times are close but not equal.
                        if clock_pin == self.clock_pin and 0 < abs(start_ns - clock_start_ns) < 60:
                            if self.instruction_conflict_resolution_method == 'abort':
                                raise ValueError(f'Conflict between flip {m_flip} and clock channel flip. Times: {start_ns} and {clock_start_ns}. Re-examine pulse timing or change instruction_conflict_resolution_method.')
                            elif self.instruction_conflict_resolution_method == 'round_early':
                                # Round to earliest time that does not conflict with the clock flip.
                                if start_ns < clock_start_ns:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns - 60
                                elif start_ns > clock_start_ns:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns
                            elif self.instruction_conflict_resolution_method == 'round_late':
                                # Round to latest time that does not conflict with the clock flip
                                if start_ns < clock_start_ns:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns
                                elif start_ns > clock_start_ns:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns + 60
                            elif self.instruction_conflict_resolution_method == 'round_near':
                                # Round to nearest time that does not conflict with the clock flip.
                                if start_ns < clock_start_ns - 30:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns - 60
                                elif start_ns > clock_start_ns + 30:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns + 60
                                else:
                                    chflip_pin_change_startns[m_flip][2] = clock_start_ns
                            else:
                                raise ValueError(f'Unknown instruction_conflict_resolution_method: {self.instruction_conflict_resolution_method}.')
                            print([
                                'Non-clock flip conflict resolved: ',
                                f'm_flip={m_flip}, pin={pin}, change={change}, start_ns={start_ns} --> {chflip_pin_change_startns[m_flip][2]}'
                                ])
                            break

        # # Clock constraint -- arb gate free falling edge
        # if self.clock_pin is not None and self.clock_type == 'arb_gate_free_falling_edge':
        #     for m_flip in range(num_channel_flips):
        #         # Get the info of the flip event that might need to be changed
        #         pin, change, start_ns = chflip_pin_change_startns[m_flip]
        #         # It will only be changed if it is not the clock pin or if it is a falling edge.
        #         if pin != self.clock_pin:
        #             for clock_chflip in chflip_pin_change_startns:
        #                 # Get the info of the flip that might be a clock channel flip.
        #                 clock_pin, clock_change, clock_start_ns = clock_chflip
        #                 # Check if it is actually a clock channel flip and if the flip times violate the PB constraints.
        #                 if clock_pin == self.clock_pin and 0 < abs(start_ns - clock_start_ns) < 60:
        #                     if clock_change == 1:
        #                         if self.instruction_conflict_resolution_method == 'abort':
        #                             raise ValueError(f'Conflict between flip {m_flip} and clock channel flip. Times: {start_ns} and {clock_start_ns}. Re-examine pulse timing or change instruction_conflict_resolution_method.')
        #                         elif self.instruction_conflict_resolution_method == 'round_early':
        #                         # Round to earliest time that does not conflict with the clock flip.
        #                             if start_ns < clock_start_ns:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns - 60
        #                             elif start_ns > clock_start_ns:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns
        #                         elif self.instruction_conflict_resolution_method == 'round_late':
        #                             # Round to latest time that does not conflict with the clock flip
        #                             if start_ns < clock_start_ns:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns
        #                             elif start_ns > clock_start_ns:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns + 60
        #                         elif self.instruction_conflict_resolution_method == 'round_near':
        #                             # Round to nearest time that does not conflict with the clock flip.
        #                             if start_ns < clock_start_ns - 30:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns - 60
        #                             elif start_ns > clock_start_ns + 30:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns + 60
        #                             else:
        #                                 chflip_pin_change_startns[m_flip][2] = clock_start_ns
        #                         else:
        #                             raise ValueError(f'Unknown instruction_conflict_resolution_method: {self.instruction_conflict_resolution_method}.')
        #                     print([
        #                             'Non-clock flip conflict resolved: ',
        #                             f'm_flip={m_flip}, pin={pin}, change={change}, start_ns={start_ns} --> {chflip_pin_change_startns[m_flip][2]}'
        #                             ])
        #                             break






        # There should now be no conflicts at any clock channel flip times if the clock is fixed frequency.
        # Check for remaining conflicts.
        m_conflict_res_loop = 0
        # This will repeat until there are no conflicts.
        while True:
            # This will count the number of times the loop has run.
            m_conflict_res_loop += 1
            # Sort the channel flips by start time.
            chflip_pin_change_startns.sort(key=lambda x: x[2])
            # This will count the number of conflicts.
            num_conflicts = 0
            # Here, adjustments are made to the next channel flip if it is too close to the current channel flip.
            # So, the loop goes up to the second-to-last channel flip.
            for m_flip in range(num_channel_flips-1):
                # Get the info of the current channel flip.
                pin, change, start_ns = chflip_pin_change_startns[m_flip]
                # Skip this iteration if the current channel flip is a clock channel flip. We know there are no conflicts at clock channel flip times.
                if pin == self.clock_pin and self.clock_type == 'fixed_frequency':
                    continue
                # Get the info of the next channel flip.
                next_pin, next_change, next_start_ns = chflip_pin_change_startns[m_flip + 1]
                # Skip this iteration if the next channel flip is a clock channel flip. We know there are no conflicts at clock channel flip times.
                if next_pin == self.clock_pin and (
                    self.clock_type == 'fixed_frequency' 
                    or (self.clock_type == 'arb_gate_free_falling_edge' and next_change == 1)
                    ):
                    continue
                # Check if the next channel flip is too close to the current channel flip.
                if next_start_ns != start_ns and next_start_ns < start_ns + 60:
                    # If it is, increment the number of conflicts.
                    num_conflicts += 1
                    if self.instruction_conflict_resolution_method == 'abort':
                        raise ValueError(f'Conflict between channel flips {m_flip} and {m_flip + 1}. Times: {start_ns} and {next_start_ns}. Pins: {pin} and {next_pin}. Re-examine pulse timing or change instruction_conflict_resolution_method.')
                    else:
                        this_resolution_method = self.instruction_conflict_resolution_method
                        if pin == next_pin:
                            this_resolution_method = 'round_late'
                        if this_resolution_method == 'round_early':
                            # Round the next channel flip to the earliest time that does not conflict with this channel flip.
                            chflip_pin_change_startns[m_flip + 1][2] = start_ns
                        elif this_resolution_method == 'round_late':
                            # Round the next channel flip to the latest time that does not conflict with this channel flip.
                            chflip_pin_change_startns[m_flip + 1][2] = start_ns + 60
                        elif this_resolution_method == 'round_near':
                            # Round the next channel flip to the nearest time that does not conflict with this channel flip.
                            if next_start_ns < start_ns + 30:
                                chflip_pin_change_startns[m_flip + 1][2] = start_ns
                            elif next_start_ns > start_ns + 30:
                                chflip_pin_change_startns[m_flip + 1][2] = start_ns + 60
                        else:
                            raise ValueError(f'Unknown instruction_conflict_resolution_method: {self.instruction_conflict_resolution_method}.')
                        print([
                            'PB channel flip conflict resolved: m_conflict_res_loop = {m_conflict_res_loop}, num_conflicts = {num_conflicts}, ',
                            f'm_flip={m_flip + 1}, pin={next_pin}, change={next_pin}, start_ns={next_start_ns} --> {chflip_pin_change_startns[m_flip + 1][2]}'
                            ])
            if num_conflicts == 0:
                # If there are no conflicts, break the while-loop.
                break
            if m_conflict_res_loop > 1000:
                raise ValueError('Too many conflict resolution loops.')

        # %%
        # Update the start_stop_actual_ns for each channel with the corrected start times
        for channel in self.active_channels:
            m_ss = 0
            for flip in chflip_pin_change_startns:
                if flip[0] == channel.pin:
                    if flip[1] == 1:
                        channel.start_stop_actual_ns[m_ss][0] = flip[2]
                    elif flip[1] == -1:
                        channel.start_stop_actual_ns[m_ss][1] = flip[2]
                        m_ss += 1

        # Create directories for pin to active channel index
        active_ch_idx_pin_directory = {channel.pin: idx for idx, channel in enumerate(self.active_channels)}
        
        # Convert to channel flips to array of instructions and durations
        # current_pin_arr is the current state of the pins this will get copied to 
        # self.instructions_pin_arr each time all simultaneous flips are accounted for in an instruction.
        current_pin_arr = np.zeros(self.num_active_channels, dtype=int)
        # self.instructions_pin_arr is an array of pin states for each instruction. These will be converted to instruction words later.
        # Initialize this with length num_channel_flips + 1, but it will be shortened if any flips are simultaneous. The +1 is for the final instruction if there is all off time.
        self.instructions_pin_arr = np.zeros((num_channel_flips + 1, self.num_active_channels), dtype=int)
        # self.instruction_durations is an array of durations for each instruction.
        self.instruction_durations = np.zeros(num_channel_flips + 1, dtype=int)
        m_instruction = 0
        if chflip_pin_change_startns[0][2] != 0:
            # If the first flip is not at time 0, then there is an instruction with all channels off.
            self.instructions_pin_arr[m_instruction] = current_pin_arr.copy()
            self.instruction_durations[m_instruction] = chflip_pin_change_startns[0][2]
            m_instruction += 1
        # Loop through the channel flips
        for m_flip in range(num_channel_flips):
            # Get the info of the flip event
            pin, change, start_ns = chflip_pin_change_startns[m_flip]
            # Flip the state of the pin
            current_pin_arr[active_ch_idx_pin_directory[pin]] += change
            if current_pin_arr[active_ch_idx_pin_directory[pin]] not in [0, 1]:
                raise ValueError(f'Invalid pin state: {current_pin_arr[active_ch_idx_pin_directory[pin]]} for pin {pin} at start_ns {start_ns}. Expected 0 or 1. Check instructions. Possible conflict resolution handling did not anticipate this case.')
            # Will this instruction flip additional channels?
            # This will only be the case if the next flip is not the final flip.
            if m_flip < num_channel_flips - 1:
                # Get the info of the next flip event.
                next_pin, next_change, next_start_ns = chflip_pin_change_startns[m_flip + 1]
                if next_start_ns == start_ns:
                    # If the next flip is at the same time as the current flip, then continue the 
                    # for-loop to the next flip without incrementing m_instruction.
                    continue
                elif next_start_ns >= start_ns + 60:
                    # If the next flip is at least 60 ns after the current flip, then
                    # the current_pin_arr is the instruction_pin_arr for this instruction.
                    self.instructions_pin_arr[m_instruction] = current_pin_arr.copy()
                    # The duration of the instruction is the time between the current flip and the next flip.
                    self.instruction_durations[m_instruction] = next_start_ns - start_ns
                    # Increment the instruction counter.
                    m_instruction += 1
                elif next_start_ns < start_ns + 60:
                    # If the next flip is less than 60 ns after the current flip, then
                    # there is a problem. This error shouldn't occur if the conflict resolution
                    # methods are working correctly.
                    raise ValueError(f'Conflict between channel flips {m_flip} and {m_flip + 1}. Times: {start_ns} and {next_start_ns}. Pins: {pin} and {next_pin}. This should not happen unless there is a problem with conflict resolution.')
                else:
                    raise ValueError('Unknown error in instruction pin array generation.')
            else:
                # If a channel has a stop time at the end of the cycle_period, then there is no flip.
                # This will result in a zero duration instruction.
                final_instruction_duration_ns = int(np.round(self.cycle_period * 1e9)) - start_ns
                if final_instruction_duration_ns >= 60:
                    # For the last flip, the current_pin_arr is the instruction_pin_arr for this instruction.
                    self.instructions_pin_arr[m_instruction] = current_pin_arr.copy()
                    self.instruction_durations[m_instruction] = final_instruction_duration_ns
                    m_instruction += 1
                elif final_instruction_duration_ns < 60 and final_instruction_duration_ns != 0:
                    raise ValueError(f'Conflict: final instruction duration is {final_instruction_duration_ns} ns. Pin: {pin}. This should not happen unless there is a problem with conflict resolution.')


        # Remove any additional unused instructions
        self.instructions_pin_arr = self.instructions_pin_arr[:m_instruction]
        self.instruction_durations = self.instruction_durations[:m_instruction]
        self.num_instructions = len(self.instruction_durations)

        # Convert the pin arrays to instruction words. Each word is an integer whose bits represent the pin states.
        self.instruction_pin_words = np.zeros(self.num_instructions, dtype=int)
        for m_instruction in range(self.num_instructions):
            # print(f'm_instruction: {m_instruction}')
            self.instruction_pin_words[m_instruction] = 0x0
            for m_channel in range(self.num_active_channels):
                # print(f'm_channel: {m_channel}')
                self.instruction_pin_words[m_instruction] += int(self.instructions_pin_arr[m_instruction,m_channel]) << self.active_channels[m_channel].pin
        
        # print(f'chflip_pin_change_startns: {[chflip for chflip in chflip_pin_change_startns]}')
        # print(f'self.instructions_pin_arr: {self.instructions_pin_arr}')
        # print(f'self.instruction_durations: {self.instruction_durations}')
        # print(f'self.instruction_pin_words: {self.instruction_pin_words}')
        

    def visualize_pb_sequence(self, time_range=None):
        import matplotlib.pyplot as plt

        real_time_channel_fig = plt.figure()

        # N = self.num_active_channels
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, self.num_active_channels))
        # colors = ['r', 'g', 'b']  # Define colors for each channel

        # Plot the intended and actual channel states
        for m_channel in range(self.num_active_channels):
            # Get the channel object
            channel = self.active_channels[m_channel]
            # times_adjusted are the times after incorporating delays and added pulse lengths.
            times_adjusted = [0]
            # times_intended are the times before incorporating delays and added pulse lengths, e.g., what the qubit experiences.
            times_intended = [0]
            # Values are the on/off values, offset by m_channel for visualization.
            values = [m_channel]
            for start_ns, stop_ns in channel.start_stop:
                # Each instruction will appear as a square wave with vertical edges.
                times_intended.extend([start_ns, start_ns, stop_ns, stop_ns])
            for start_ns, stop_ns in channel.start_stop_actual_ns:
                times_adjusted.extend([start_ns, start_ns, stop_ns, stop_ns])
                # The values are the on/off values. Here they are offset by m_channel 
                # and scaled by 0.8 to avoid overlapping visuals.
                values.extend([m_channel, m_channel + 0.8, m_channel + 0.8, m_channel])
            # Add the final cycle_period instruction. 
            # Adjusted times are in nanoseconds, so convert to s.
            times_adjusted.extend([self.cycle_period * 1e9])
            times_intended.extend([self.cycle_period])
            values.extend([m_channel])
            
            plt.plot(np.array(times_adjusted) * 1e-9, values, label=f'pin {channel.pin}, {channel.channel_name} adj.', linestyle='--', color=colors[m_channel])
            plt.plot(np.array(times_intended), values, label=f'pin {channel.pin}, {channel.channel_name} int.', linestyle='-', color=colors[m_channel])
        
        if time_range is not None:
            plt.xlim(time_range)
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('PB output -- from start/stop times')
        # XL = plt.xlim()
        # plt.xlim([XL[0], XL[1]*1.5])
        handles, labels = plt.gca().get_legend_handles_labels()
        order = range(len(labels)-1, -1, -1)
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.7)  # Add extra space on the right for the legend
        plt.gcf().set_size_inches(12, 8)  # Increase the width of the figure window
        plt.show(block=False)
        plt.pause(0.001)

        # Additional figure straight from the pin array.
        pin_arr_to_realtime_fig = plt.figure()

        # The time coordinates are given by the cumulative sum of the instruction durations.
        output_times = np.cumsum(self.instruction_durations)
        # Repeat each time twice for the vertical edges of the square wave.
        output_times = np.repeat(output_times, 2)
        # Add a zero at the beginning to show initial state.
        output_times = np.insert(output_times, [0,0], 0)
        
        # Visual pin array
        visual_pin_arr = self.instructions_pin_arr.copy()
        # Repeat each pin twice for the vertical edges of the square wave.
        visual_pin_arr = np.repeat(visual_pin_arr, 2, axis=0)
        # Add a row of zeros at the beginning to show switching on.
        visual_pin_arr = np.vstack([np.zeros(self.num_active_channels), visual_pin_arr])
        # Add a row of zeros at the end to show switching off.
        visual_pin_arr = np.vstack([visual_pin_arr, np.zeros(self.num_active_channels)])
        # Convert to float for plotting.
        visual_pin_arr = np.array(visual_pin_arr, dtype=float)

        for m_channel in range(self.num_active_channels):
            plt.plot(
                output_times * 1e-9, 
                0.8*visual_pin_arr[:, m_channel] + m_channel, 
                label=f'pin {self.active_channels[m_channel].pin}, {self.active_channels[m_channel].channel_name}',
                color=colors[m_channel]
            )
        
        if time_range is not None:
            plt.xlim(time_range)
        plt.grid()
        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0), useMathText=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('PB output -- from pin array')
        handles, labels = plt.gca().get_legend_handles_labels()
        order = range(len(labels)-1, -1, -1)
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.7)  # Add extra space on the right for the legend
        plt.gcf().set_size_inches(12, 8)  # Increase the width of the figure window
        plt.show(block=False)
        plt.pause(0.001)

        return real_time_channel_fig, pin_arr_to_realtime_fig

    # def program_pb_loop_with_alloffs_and_run(
    #         self,
    #         check_visualization = True,
    #         number_of_loop_rpts = None,
    #         all_off_duration_ns = None):
    #     # %% Program the PulseBlaster
        
    #     if check_visualization:
    #         self.visualize_pb_sequence()
    #         input('Press Enter to program the PulseBlaster.')
        
    #     import qdlutils.hardware.pulsers.PulseBlaster.pulseblasterinterface as pulseblasterinterface
    #     pbi = pulseblasterinterface.PulseBlasterInterface()
    #     pbi.pb_board_number = 1

    #     import qdlutils.hardware.pulsers.PulseBlaster.spinapi as pb_spinapi
        
    #     pbi.stop()
    #     pbi.open()
    #     pbi.start_programming()

    #     # Pre-loop all off.
    #     pb_spinapi.pb_inst_pbonly(
    #         0x0,
    #         pb_spinapi.Inst.CONTINUE,
    #         0,
    #         all_off_duration_ns
    #     )

    #     # Start of loop
    #     m_instr = 0
    #     if number_of_loop_rpts == np.inf:
    #         loop_start_instruction = pb_spinapi.Inst.CONTINUE
    #         loop_start_num_loops_arg = 0
    #     else:
    #         loop_start_instruction = pb_spinapi.Inst.LOOP
    #         loop_start_num_loops_arg = number_of_loop_rpts
    #     loop_start = pb_spinapi.pb_inst_pbonly(
    #         int(self.instruction_pin_words[m_instr]),
    #         loop_start_instruction,
    #         loop_start_num_loops_arg,
    #         int(self.instruction_durations[m_instr])
    #     )

    #     # All intermediate instructions
    #     for m_instr in range(1, self.num_instructions-1):
    #         pb_spinapi.pb_inst_pbonly(
    #             int(self.instruction_pin_words[m_instr]),
    #             pb_spinapi.Inst.CONTINUE,
    #             0,
    #             int(self.instruction_durations[m_instr])
    #         )

    #     # End of the loop
    #     m_instr = self.num_instructions - 1
    #     if number_of_loop_rpts == np.inf:
    #         loop_end_instruction = pb_spinapi.Inst.BRANCH
    #     else:
    #         loop_end_instruction = pb_spinapi.Inst.END_LOOP
    #     pb_spinapi.pb_inst_pbonly(
    #         int(self.instruction_pin_words[m_instr]),
    #         loop_end_instruction,
    #         loop_start,
    #         int(self.instruction_durations[m_instr])
    #     )

    #     # Post-loop all off.
    #     pb_spinapi.pb_inst_pbonly(
    #         0x0,
    #         pb_spinapi.Inst.STOP,
    #         0,
    #         200
    #     )

    #     pbi.stop_programming()

    #     pbi.run_the_pb_sequence()




