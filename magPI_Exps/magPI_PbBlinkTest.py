# %%

from qdlutils.hardware.pulsers.pulseblaster import PulseBlasterBlinkTest

pbblink = PulseBlasterBlinkTest(
    blink_channels = [0],
    cycle_period_s = 1.0,
    duty_cycle = 0.25,
    num_loops = 5)
pbblink.run()

# %%
print(f'pbblink.on_time_ns = {pbblink.on_time_ns}')
print(f'pbblink.off_time_ns = {pbblink.off_time_ns}')

# %%
import numpy as np
import matplotlib.pyplot as plt
import time

import logging

# %%
import qdlutils.experiments.podmr
from qdlutils.hardware.pulsers.pulseblaster import PulseBlasterPulsedODMR
from qdlutils.experiments.rabi import signal_to_background
# import qdlutils.analysis.aggregation  # DOES NOT EXIST....?
import qdlutils.hardware.nidaq
import qt3rfsynthcontrol

# %%
from qdlutils.hardware.pulsers.pulseblaster import PulseBlasterHoldAOM
hold_aom_pulser = PulseBlasterHoldAOM(
    pb_board_number = 1,
    aom_channel = 0,
    cycle_width = 10e-3
    )
hold_aom_pulser.program_pulser_state()
hold_aom_pulser.start()


# %%
# import sys
# # sys.path.append(".")
# print(sys.path)
# import os
# print(os.getcwd())
# import magPI.magPI_Python.Settings.magPI_config as magPI_config

# %%
logging.basicConfig(level=logging.WARNING)

# %%
qdlutils.experiments.podmr.logger.setLevel(logging.INFO)

# %%
rfsynth = qt3rfsynthcontrol.QT3SynthHD('COM7')

# %%
nidaq_config = qdlutils.hardware.nidaq.EdgeCounter('Dev1')

# %%
podmr_pulser = PulseBlasterPulsedODMR(
    pb_board_number = 1, 
    aom_channel = 0,
    rf_channel = 1,
    clock_channel = 5,
    trigger_channel = 2)

    # clock_period = 200e-9,
    # trigger_width = 500e-9,
    # rf_pulse_duration = 2e-6,
    # aom_width=5e-6,
    # aom_response_time = 800e-9,
    # rf_response_time = 200e-9,
    # pre_rf_pad = 100e-9,
    # post_rf_pad = 100e-9,
    # full_cycle_width = 30e-6,

    # rf_pulse_justify = 'center')

# %%
active_pins = [0, 1, 5, 8]
on_word = 0x0
for pin in active_pins:
    on_word |= 1 << pin

print(bin(on_word))
print(on_word)
print(hex(on_word))
# print(0x10E)
# print(bin(0x10E))
# print(int('0b100001110', 2))
# print(int('0x10E', 16))

# %%
pb_blink = 

# %%
print(['0'*4]*3)

# %%
import numpy as np
cycle_period_s = 0.2151984
# cycle_period_ns = int(np.round(cycle_period_s/(50e-9))*50e-9)
# print(cycle_period_ns)

a = int(cycle_period_s / 50e-9) * 50
b = int(np.round(cycle_period_s/(50e-9))*50)
print(a)
print(b)
print(a-b)
if 1>0:
    raise MyError('basic error')

# %%
#from rabi osc experiment
#for rf_power = -25, pi pulse width = 1e-6
experiment = qdlutils.experiments.podmr.PulsedODMR(
    podmr_pulser, rfsynth, nidaq_config, 
    photon_counter_nidaq_terminal = 'PFI0',
    clock_nidaq_terminal = 'PFI12',
    trigger_nidaq_terminal = 'PFI13',
    freq_low = 2820e6,
    freq_high = 2920e6,
    freq_step = 1e6,
    rf_power = -40,
    rfsynth_channel = 0)

# %%
experiment.N_clock_ticks_per_cycle

# %%
scan_data = experiment.run(N_cycles=100000)

# scan_data is a list where each element of the list is the result of the CWODMR data acquired at a particular frequency.

# Each element of scan_data contains a list of size two. The first element is the frequency, the second element is 
# an array of data that is the output of the aggregation of the full raw data stream. The size of this array of data
# will be equal to cwodmr.N_clock_ticks_per_cycle

# scan_data - [
#   [2700e6, [d_1, d_2, d_3, ... d_N_clock_ticks_per_cycle]],
#   [2705e6, [d_1, d_2, d_3, ... d_N_clock_ticks_per_cycle]]
#   ...
#   [3000e6, [d_1, d_2, d_3, ... d_N_clock_ticks_per_cycle]]

# ]


