import importlib
import importlib.resources
import logging
import numpy as np
import datetime
import h5py

from threading import Thread
import tkinter as tk
import yaml

from scipy.optimize import curve_fit

import qdlutils
from qdlutils.applications.qdlscan.application_controller import ScanController
from qdlutils.applications.qdlscan.application_gui import (
    LauncherApplicationView,
    LineScanApplicationView,
    ImageScanApplicationView
)
import qdlutils.applications.qdlscope.main as qdlscope

logger = logging.getLogger(__name__)
logging.basicConfig()


CONFIG_PATH = 'qdlutils.applications.qdlscan.config_files'
DEFAULT_CONFIG_FILE = 'qdlscan_base.yaml'

# Dictionary for converting axis to an index
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

# Default color map
DEFAULT_COLOR_MAP = 'gray'


class LauncherApplication:
    '''
    This is the launcher class for the `qdlscan` application which handles the 
    initalization of individual scanning confocal measurements. This application is
    serves as a replacement of `qt3scan` which relies on some of the older architecture
    for DAQ interfacing. The overall application structure is described below:

    The central class of the application is the `ScanController` which is located in
    `qdlscan/application_controller.py`. This class manages the hardware, which is by
    default NIDAQ-controlled piezos (`NidaqPositionController` in 
    `qdlutils/hardware/nidaq/analogoutputs/nidaqposition.py`) and NIDAQ edge counter
    (`NidaqTimedRateCounter` in `qdlutils/hardware/nidaq/counters/nidaqtimedratecounter.py`)
    however a suitably designed alternative class for other hardware could also be
    reasonably used without much effort.

    `ScanController` handles the actual movement of the piezos and coordinates it with
    the counters in order to perform basic confocal scanning measurements. The 
    application launches and instantiates a single `ScanController`, dubbed the 
    "application controller", which it uses to perform scans. The application 
    controller can be queried to move the piezos or perform specifc scans. Since it is
    shared by all scans performed in the application session, it has built-in logic to
    ignore calls when it is busy, aided by threading. If one desires to perform 
    confocal scanning is appreciably different hardware, then the scan controller
    could also be rewritten to interface with it if the current implementation does
    not immediately work.

    The `LauncherApplication` is the other main class. It instantiates the `ScanController`
    and creates a GUI (`LauncherApplicationView`) which allows the user to input scan
    parameters and to launch image and line scans, as well as move the piezos directly.

    When an image or line scan is launched, the parameters are read out from the GUI and
    a new instance of the `ImageScanApplication` or `LineScanApplication` are created
    in each case respectively. These sub-applications contain metadata and actual 
    measured data for the given scan (or set of scans) that they represent. On
    initialization, they comppute (from the GUI inputs) information to setup the scan(s),
    making adjustments if needed, and then begin scanning in a thread. After, or between,
    scans, these sub-applications store the results and update the figure. Additionally,
    these sub-applications also handle the saving, pausing, and continuing of scans where
    applicable.

    For additional details about the individual components please see their docstrings.
    '''

    def __init__(self, default_config_filename: str, is_root_process: bool) -> None:
        '''
        Initialization for the LauncherApplication class. It loads the application
        controller and various hardware, then creates a GUI and binds the buttons.
        Callback methods for the GUI interactions are contained in this class.
        
        Parameters
        ----------
        default_config_filename: str
            Filename of the default config YAML file. It must be located in the
            `qdlscan/config_files` directory.
        '''
        # Boolean if the function is the root or not, determines if the application is
        # intialized via tk.Tk or tk.Toplevel
        self.is_root_process = is_root_process

        # Attributes
        self.application_controller = None
        self.min_x_position = None
        self.min_y_position = None
        self.min_z_position = None
        self.max_x_position = None
        self.max_y_position = None
        self.max_z_position = None
        self.max_x_range = None
        self.max_y_range = None
        self.max_z_range = None

        # Number of scan windows launched
        self.number_scans = 0
        # Most recent scan -- maybe not needed?
        self.current_scan = None
        # Dictionary of scan parameters (from control gui)
        self.scan_parameters = None
        # Dictionary of daq parameters (from control gui)
        self.daq_parameters = None

        # Last save directory
        self.last_save_directory = None

        # Load the YAML file
        self.load_yaml_from_name(yaml_filename=default_config_filename)

        # Initialize the root tkinter widget (window housing GUI)
        if self.is_root_process:
            self.root = tk.Tk()
        else:
            self.root = tk.Toplevel()
        # Create the main application GUI
        self.view = LauncherApplicationView(main_window=self.root)

        # Bind the buttons
        self.view.control_panel.image_start_button.bind('<Button>', self.start_image_scan)
        self.view.control_panel.line_start_x_button.bind('<Button>', self.optimize_x_axis)
        self.view.control_panel.line_start_y_button.bind('<Button>', self.optimize_y_axis)
        self.view.control_panel.line_start_z_button.bind('<Button>', self.optimize_z_axis)
        self.view.control_panel.x_axis_set_button.bind('<Button>', self.set_x_axis)
        self.view.control_panel.y_axis_set_button.bind('<Button>', self.set_y_axis)
        self.view.control_panel.z_axis_set_button.bind('<Button>', self.set_z_axis)
        self.view.control_panel.get_position_button.bind('<Button>', self.get_coordinates)
        self.view.control_panel.open_counter_button.bind('<Button>', self.open_counter)

    def run(self) -> None:
        '''
        This function launches the application including the GUI
        '''
        # Set the title of the app window
        self.root.title("qdlscan")
        # Display the window (not in task bar)
        self.root.deiconify()
        # Launch the main loop
        if self.is_root_process:
            self.root.mainloop()

    def configure_from_yaml(self, afile: str) -> None:
        '''
        This method loads a YAML file to configure the qdlmove hardware
        based on yaml file indicated by argument `afile`.

        This method instantiates and configures the controllers and counters
        for the scan application, then creates the application controller.

        Parameters
        ----------
        afile: str
            Full-path filename of the YAML config file.
        '''
        with open(afile, 'r') as file:
            # Log selection
            logger.info(f"Loading settings from: {afile}")
            # Get the YAML config as a nested dict
            config = yaml.safe_load(file)

        # First we get the top level application name
        APPLICATION_NAME = list(config.keys())[0]

        # Get the names of the counter and positioners
        hardware_dict = config[APPLICATION_NAME]['ApplicationController']['hardware']
        counter_name = hardware_dict['counter']
        x_axis_name = hardware_dict['x_axis_control']
        y_axis_name = hardware_dict['y_axis_control']
        z_axis_name = hardware_dict['z_axis_control']

        # Get the counter, instantiate, and configure
        import_path = config[APPLICATION_NAME][counter_name]['import_path']
        class_name = config[APPLICATION_NAME][counter_name]['class_name']
        module = importlib.import_module(import_path)
        logger.debug(f"Importing {import_path}")
        constructor = getattr(module, class_name)
        counter = constructor()
        counter.configure(config[APPLICATION_NAME][counter_name]['configure'])

        # Get the x axis instance
        import_path = config[APPLICATION_NAME][x_axis_name]['import_path']
        class_name = config[APPLICATION_NAME][x_axis_name]['class_name']
        module = importlib.import_module(import_path)
        logger.debug(f"Importing {import_path}")
        constructor = getattr(module, class_name)
        x_axis = constructor()
        x_axis.configure(config[APPLICATION_NAME][x_axis_name]['configure'])
        # Get the limits
        self.min_x_position = config[APPLICATION_NAME][x_axis_name]['configure']['min_position']
        self.max_x_position = config[APPLICATION_NAME][x_axis_name]['configure']['max_position']
        self.max_x_range = self.max_x_position - self.min_x_position

        # Get the y axis instance
        import_path = config[APPLICATION_NAME][y_axis_name]['import_path']
        class_name = config[APPLICATION_NAME][y_axis_name]['class_name']
        module = importlib.import_module(import_path)
        logger.debug(f"Importing {import_path}")
        constructor = getattr(module, class_name)
        y_axis = constructor()
        y_axis.configure(config[APPLICATION_NAME][y_axis_name]['configure'])
        # Get the limits
        self.min_y_position = config[APPLICATION_NAME][y_axis_name]['configure']['min_position']
        self.max_y_position = config[APPLICATION_NAME][y_axis_name]['configure']['max_position']
        self.max_y_range = self.max_y_position - self.min_y_position

        # Get the z axis instance
        import_path = config[APPLICATION_NAME][z_axis_name]['import_path']
        class_name = config[APPLICATION_NAME][z_axis_name]['class_name']
        module = importlib.import_module(import_path)
        logger.debug(f"Importing {import_path}")
        constructor = getattr(module, class_name)
        z_axis = constructor()
        z_axis.configure(config[APPLICATION_NAME][z_axis_name]['configure'])
        # Get the limits
        self.min_z_position = config[APPLICATION_NAME][z_axis_name]['configure']['min_position']
        self.max_z_position = config[APPLICATION_NAME][z_axis_name]['configure']['max_position']
        self.max_z_range = self.max_x_position - self.min_x_position

        # Get the application controller constructor 
        import_path = config[APPLICATION_NAME]['ApplicationController']['import_path']
        class_name = config[APPLICATION_NAME]['ApplicationController']['class_name']
        module = importlib.import_module(import_path)
        logger.debug(f"Importing {import_path}")
        constructor = getattr(module, class_name)
        # Get the configure dictionary
        controller_config_dict = config[APPLICATION_NAME]['ApplicationController']['configure']
        # Create the application controller passing the hardware and the config dict
        # as kwargs.
        self.application_controller = constructor(
            **{'x_axis_controller': x_axis,
               'y_axis_controller': y_axis,
               'z_axis_controller': z_axis,
               'counter_controller': counter,
               **controller_config_dict}
        )

    def load_yaml_from_name(self, yaml_filename: str) -> None:
        '''
        Loads the yaml configuration file from name.

        Should be called during instantiation of this class and should be the callback
        function for loading of other standard yaml files while running.

        Parameters
        ----------
        yaml_filename: str
            Filename of the .yaml file in the qdlscan/config_files path.
        '''
        yaml_path = importlib.resources.files(CONFIG_PATH).joinpath(yaml_filename)
        self.configure_from_yaml(str(yaml_path))

    def enable_buttons(self) -> None:
        pass
    def disable_buttons(self) -> None:
        pass

    def optimize_x_axis(self, tkinter_event=None) -> None:
        '''
        Callback to optimize X button in launcher GUI.
        Gets the config data written in the GUI. If it is valid then it will create an 
        instnace of the `LineScanApplication` class to scan the axis.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        if self.application_controller.busy:
            logger.error(f'Application controller is current busy.')
            return None
        logger.info('Optimizing X axis.')
        # Update the parameters
        try:
            self._get_scan_config()
        except Exception as e:
            logger.error(f'Scan parameters are invalid: {e}')
            return None
        # Increase the nunmber of scans launched
        self.number_scans += 1
        # Launch a line scan application
        self.current_scan = LineScanApplication(
            parent_application = self,
            application_controller = self.application_controller,
            axis = 'x',
            range = self.scan_parameters['line_range_xy'],
            n_pixels = self.scan_parameters['line_pixels'],
            time = self.scan_parameters['line_time'],
            id = str(self.number_scans).zfill(3)
        )

    def optimize_y_axis(self, tkinter_event=None) -> None:
        '''
        Callback to optimize Y button in launcher GUI.
        Gets the config data written in the GUI. If it is valid then it will create an 
        instnace of the `LineScanApplication` class to scan the axis.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        if self.application_controller.busy:
            logger.error(f'Application controller is current busy.')
            return None
        logger.info('Optimizing Y axis.')
        # Update the parameters
        try:
            self._get_scan_config()
        except Exception as e:
            logger.error(f'Scan parameters are invalid: {e}')
            return None
        # Increase the nunmber of scans launched
        self.number_scans += 1
        # Launch a line scan application
        self.current_scan = LineScanApplication(
            parent_application = self,
            application_controller = self.application_controller,
            axis = 'y',
            range = self.scan_parameters['line_range_xy'],
            n_pixels = self.scan_parameters['line_pixels'],
            time = self.scan_parameters['line_time'],
            id = str(self.number_scans).zfill(3)
        )

    def optimize_z_axis(self, tkinter_event=None) -> None:
        '''
        Callback to optimize Z button in launcher GUI.
        Gets the config data written in the GUI. If it is valid then it will create an 
        instnace of the `LineScanApplication` class to scan the axis.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        if self.application_controller.busy:
            logger.error(f'Application controller is current busy.')
            return None
        logger.info('Optimizing Z axis.')
        # Update the parameters
        try:
            self._get_scan_config()
        except Exception as e:
            logger.error(f'Scan parameters are invalid: {e}')
            return None
        # Increase the nunmber of scans launched
        self.number_scans += 1
        # Launch a line scan application
        self.current_scan = LineScanApplication(
            parent_application = self,
            application_controller = self.application_controller,
            axis = 'z',
            range = self.scan_parameters['line_range_z'],
            n_pixels = self.scan_parameters['line_pixels'],
            time = self.scan_parameters['line_time'],
            id = str(self.number_scans).zfill(3)
        )

    def start_image_scan(self, tkinter_event=None) -> None:
        '''
        Callback to start scan button in launcher GUI.
        Gets the config data written in the GUI. If it is valid then it will create an 
        instnace of the `ImageScanApplication` class to create a confocal scan image.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        if self.application_controller.busy:
            logger.error(f'Application controller is current busy.')
            return None
        logger.info('Starting confocal image scan.')
        # Update the parameters
        try:
            self._get_scan_config()
        except Exception as e:
            logger.error(f'Scan parameters are invalid: {e}')
            return None
        # Increase the nunmber of scans launched
        self.number_scans += 1
        # Launch a image scan application
        self.current_scan = ImageScanApplication(
            parent_application = self,
            application_controller = self.application_controller,
            axis_1 = 'x',
            axis_2 = 'y',
            range = self.scan_parameters['image_range'],
            n_pixels = self.scan_parameters['image_pixels'],
            time = self.scan_parameters['image_time'],
            id = str(self.number_scans).zfill(3)
        )
    
    def set_x_axis(self, tkinter_event=None) -> None:
        '''
        Callback to set X button in launcher GUI.
        Sets the axis position to what is entered in the GUI if valid.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        try:
            self._get_daq_config()
            position = self.daq_parameters['x_position']
            self.application_controller.set_axis(axis='x', position=position)
            logger.info(f'Set x axis to {position}')
        except Exception as e:
            logger.error(f'Error with setting x axis: {e}')

    def set_y_axis(self, tkinter_event=None) -> None:
        '''
        Callback to set Y button in launcher GUI.
        Sets the axis position to what is entered in the GUI if valid.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        try:
            self._get_daq_config()
            position = self.daq_parameters['y_position']
            self.application_controller.set_axis(axis='y', position=position)
            logger.info(f'Set y axis to {position}')
        except Exception as e:
            logger.error(f'Error with setting y axis: {e}')
        
    def set_z_axis(self, tkinter_event=None) -> None:
        '''
        Callback to set Z button in launcher GUI.
        Sets the axis position to what is entered in the GUI if valid.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''
        try:
            self._get_daq_config()
            position = self.daq_parameters['z_position']
            self.application_controller.set_axis(axis='z', position=position)
            logger.info(f'Set z axis to {position}')
        except Exception as e:
            logger.error(f'Error with setting z axis: {e}')
        
    def get_coordinates(self, tkinter_event=None) -> None:
        '''
        Callback to get position button in launcher GUI.
        Populates the GUI DAQ position entries with the current position.

        Parameters
        ----------
        tkinter_event: tk.Event
            The button press event, not used.
        '''    
        x,y,z = self.application_controller.get_position()
        self.view.control_panel.x_axis_set_entry.delete(0, 'end')
        self.view.control_panel.y_axis_set_entry.delete(0, 'end')
        self.view.control_panel.z_axis_set_entry.delete(0, 'end')
        self.view.control_panel.x_axis_set_entry.insert(0, x)
        self.view.control_panel.y_axis_set_entry.insert(0, y)
        self.view.control_panel.z_axis_set_entry.insert(0, z)

    def open_counter(self, tkinter_event=None) -> None:
        try:
            qdlscope.main(is_root_process=False)
        except Exception as e:
            logger.warning(f'{e}')

    def _get_scan_config(self) -> None:
        '''
        Gets the scan parameters in the GUI and validates if they are allowable. Then 
        saves the GUI input to the launcher application if valid.
        '''
        image_range = float(self.view.control_panel.image_range_entry.get())
        image_pixels = int(self.view.control_panel.image_pixels_entry.get())
        image_time = float(self.view.control_panel.image_time_entry.get())
        line_range_xy = float(self.view.control_panel.line_range_xy_entry.get())
        line_range_z = float(self.view.control_panel.line_range_z_entry.get())
        line_pixels = int(self.view.control_panel.line_pixels_entry.get())
        line_time = float(self.view.control_panel.line_time_entry.get())

        # Check image range
        if image_range < 0.1:
            raise ValueError(f'Requested scan range {image_range} < 100 nm is too small.')
        if image_range > self.max_x_range:
            raise ValueError(f'Requested image scan range {image_range}'
                             +f' exceeds the x limit {self.max_x_range}.')
        if image_range > self.max_y_range:
            raise ValueError(f'Requested image scan range {image_range}'
                             +f' exceeds the y limit {self.max_x_range}.')
        if image_range > self.max_z_range:
            raise ValueError(f'Requested image scan range {image_range}'
                             +f' exceeds the z limit {self.max_x_range}.')
        # Check the image pixels
        if image_pixels < 1:
            raise ValueError(f'Requested image pixels {image_pixels} < 1 is too small.')
        # check the image time
        if image_time < 0.001:
            raise ValueError(f'Requested image scan time {image_time} < 1 ms is too small.')

        # Check the line xy range
        if line_range_xy < 0.1:
            raise ValueError(f'Requested scan range {line_range_xy} < 100 nm is too small.')
        if line_range_xy > self.max_x_range:
            raise ValueError(f'Requested xy scan range {line_range_xy}'
                             +f' exceeds the x limit {self.max_x_range}.')
        if line_range_xy > self.max_y_range:
            raise ValueError(f'Requested xy scan range {line_range_xy}'
                             +f' exceeds the y limit {self.max_y_range}.')
        # check the line z range
        if line_range_z < 0.1:
            raise ValueError(f'Requested scan range {line_range_z} < 100 nm is too small.')
        if line_range_z > self.max_z_range:
            raise ValueError(f'Requested z scan range {line_range_z}'
                             +f' exceeds the z limit {self.max_z_range}.')
        # Check the line pixels
        if line_pixels < 1:
            raise ValueError(f'Requested line pixels {line_pixels} < 1 is too small.')
        # check the line time
        if line_time < 0.001:
            raise ValueError(f'Requested line scan time {line_time} < 1 ms is too small.')
        if line_time > 300:
            raise ValueError(f'Requested line scan time {line_time} > 5 min is too long.')

        # Write to application memory
        self.scan_parameters = {
            'image_range': image_range,
            'image_pixels': image_pixels,
            'image_time': image_time,
            'line_range_xy': line_range_xy,
            'line_range_z': line_range_z,
            'line_pixels': line_pixels,
            'line_time': line_time,
        }

    def _get_daq_config(self) -> None:
        '''
        Gets the position parameters in the GUI and validates if they are allowable. 
        Then saves the GUI input to the launcher application if valid.
        '''
        x_position = float(self.view.control_panel.x_axis_set_entry.get())
        y_position = float(self.view.control_panel.y_axis_set_entry.get())
        z_position = float(self.view.control_panel.z_axis_set_entry.get())

        if (x_position < self.min_x_position) or (x_position > self.max_x_position):
            raise ValueError(f'Requested x coordinate {x_position} is out of bounds.')
        if (y_position < self.min_y_position) or (y_position > self.max_y_position):
            raise ValueError(f'Requested y coordinate {y_position} is out of bounds.')
        if (z_position < self.min_z_position) or (z_position > self.max_z_position):
            raise ValueError(f'Requested z coordinate {z_position} is out of bounds.')

        # Write to application memory
        self.daq_parameters = {
            'x_position': x_position,
            'y_position': y_position,
            'z_position': z_position
        }


class LineScanApplication:
    '''
    This is the line scan application class for `qdlscan` which manages the data
    and GUI output for a single scan along a particular axis. It is meant to handle
    1-d confocal scans for position optimization, although it could be made configurable.
    '''

    def __init__(self, 
                 parent_application: LauncherApplication,
                 application_controller: ScanController,
                 axis: str,
                 range: float,
                 n_pixels: int,
                 time: float,
                 id: str):
        
        self.parent_application = parent_application
        self.application_controller = application_controller
        self.axis = axis
        self.range = range
        self.n_pixels = n_pixels
        self.time = time
        self.rclick_mpl_position_x = None
        self.rclick_mpl_position_y = None

        # Time per pixel
        self.time_per_pixel = time / n_pixels

        # Optimization type
        self.optimization_method = 'gaussian'

        self.id = id
        self.timestamp = datetime.datetime.now()

        # Get the limits of the position for the axis
        if axis == 'x':
            min_allowed_position = self.parent_application.min_x_position
            max_allowed_position = self.parent_application.max_x_position
        elif axis == 'y':
            min_allowed_position = self.parent_application.min_y_position
            max_allowed_position = self.parent_application.max_y_position
        elif axis == 'z':
            min_allowed_position = self.parent_application.min_z_position
            max_allowed_position = self.parent_application.max_z_position
        else:
            raise ValueError(f'Requested axis {axis} is invalid.')

        # Get the starting position
        self.start_position_vector = application_controller.get_position()
        self.start_position_axis = self.start_position_vector[AXIS_INDEX[axis]]
        self.final_position_axis = None
        # Get the limits of the scan
        self.min_position = self.start_position_axis - (range/2)
        self.max_position = self.start_position_axis + (range/2)
        # Check if the limits exceed the range and shift range to edge
        if self.min_position < min_allowed_position:
            # Too close to minimum edge, need to shift up
            logger.warning('Start position too close to edge, shifting.')
            shift = min_allowed_position - self.min_position
            self.min_position += shift
            self.max_position += shift
        if self.max_position > max_allowed_position:
            # Too close to minimum edge, need to shift up
            logger.warning('Start position too close to edge, shifting.')
            shift = max_allowed_position - self.max_position
            self.min_position += shift
            self.max_position += shift
        # Get the scan positions (along whatever axis is being scanned)
        # We're brute forcing it here as the application controller might not sample
        # positions in the exact same way...
        self.data_x = np.linspace(start=self.min_position, 
                                  stop=self.max_position, 
                                  num=n_pixels)
        # To hold scan results
        self.data_y = np.empty(shape=n_pixels)
        self.data_y[:] = np.nan

        # Launch the line scan GUI
        # Then initialize the GUI
        self.root = tk.Toplevel()
        self.root.title(f'Scan {id} ({self.timestamp.strftime("%Y-%m-%d %H:%M:%S")})')
        self.view = LineScanApplicationView(window=self.root, 
                                            application=self,
                                            settings_dict=parent_application.scan_parameters)

        # Bind the buttons
        self.view.control_panel.save_button.bind("<Button>", self.save_scan)

        # Setup the callback for rightclicks on the figure canvas
        self.view.data_viewport.canvas.mpl_connect('button_press_event', 
                                                   lambda event: self.open_rclick(event) if event.button == 3 else None)
        # Set the commands for the right click
        self.view.rclick_menu.add_command(label='Go to position', command=self.rclick_go_to) 
        self.view.rclick_menu.add_separator() 
        self.view.rclick_menu.add_command(label='Open counter', command=self.rclick_open_counter) 

        # Launch the thread
        self.scan_thread = Thread(target=self.scan_thread_function)
        self.scan_thread.start()

    def scan_thread_function(self) -> None:
        try:
            logger.info('Starting scan thread.')
            logger.info(f'Starting scan on axis {self.axis}')
            # Run the scan (returns raw counts)
            self.data_y = self.application_controller.scan_axis(
                axis = self.axis,
                start = self.min_position,
                stop = self.max_position,
                n_pixels = self.n_pixels,
                scan_time = self.time
            )
            # Normalize to counts per second
            self.data_y = self.data_y / self.time_per_pixel
            # Optimize the position
            self.update_position()
            # Update the viewport
            self.view.update_figure()

            logger.info('Scan complete.')
        except Exception as e:
            logger.info(e)
        # Enable the buttons
        self.parent_application.enable_buttons()

    def update_position(self) -> None:
        '''
        A function to update the position of the piezos after a scan has been completed.
        Can eventually setup for multiple optimization techniques/methods.
        '''
        # Default behavior
        if self.optimization_method == 'none':
            # Do nothing
            self._optimization_method_none()
        # Fit the data to a gaussian and move to peak
        elif self.optimization_method == 'gaussian':
            # Gaussian optimize
            self._optimization_method_gaussian()
        # Move to location with highest density?
        elif self.optimization_method == 'density':
            pass

        # Move to optmial position
        self.application_controller.set_axis(axis=self.axis, position=self.final_position_axis)

    def open_rclick(self, mpl_event : tk.Event = None):
        # Get the right click position in the matplotlib axis
        self.rclick_mpl_position_x = mpl_event.xdata
        self.rclick_mpl_position_y = mpl_event.ydata
        # Get the tkinter x,y coordinates of the mouse
        mouse_x, mouse_y = self.root.winfo_pointerxy()
        # Open the popup menu with commands defined in the GUI file
        try: 
            self.view.rclick_menu.tk_popup(mouse_x,mouse_y) 
        finally: 
            self.view.rclick_menu.grab_release()
        
    def rclick_go_to(self):
        # Make sure that the position of the right click is on the canvas
        if self.rclick_mpl_position_x:
            try:
                # Go to the position on the voltage controller
                self.application_controller.set_axis(axis=self.axis, position=self.rclick_mpl_position_x)
                # Set the final position
                self.final_position_axis = self.rclick_mpl_position_x
                # Update the figure
                self.view.update_figure()
            except Exception as e:
                logger.warning(f'Error in moving to position: {e}')
        else:
            logger.warning('Right click position out of axis.')

    def rclick_open_counter(self):
        try:
            qdlscope.main(is_root_process=False)
        except Exception as e:
            logger.warning(f'{e}')
  
    def _optimization_method_none(self) -> None:
        '''
        Internal method for determining the optimal position.

        For this method we do nothing and just reset the position to the start postion.
        '''
        self.final_position_axis = self.start_position_axis

    def _optimization_method_gaussian(self) -> None:
        '''
        Internal method for determining the optimal position.

        For this method we fit the data to a Gaussian envelope centered at the maximum
        counts position.
        '''

        def fit_function(x, a, x0, sigma, c):
            return a * np.exp(-(x-x0)**2 / (2*sigma**2)) + c
        
        # Initial guess for parameters
        max_counts_idx = np.argmax(self.data_y)
        max_counts = self.data_y[max_counts_idx]
        position_max_counts = self.data_x[max_counts_idx]
        min_counts = np.min(self.data_y)
        a = max_counts - min_counts
        x0 = position_max_counts
        sigma = 0.300 # 300 microns
        c = min_counts

        try:
            p, _ = curve_fit(f=fit_function, 
                             xdata=self.data_x, 
                             ydata=self.data_y,
                             p0=[a, x0, sigma, c],
                             bounds=[[100,self.min_position, 0.250, 0],
                                     [np.inf, self.max_position, np.inf, np.inf]]) 
                                    # ^ Increase the lower bound on a to prevent fitting to noise
            
            self.final_position_axis = p[1] # Set to x0

        except Exception as e:
            logger.info('Failed to find maximum: ' + str(e))
            self.final_position_axis = self.start_position_axis
        
        
    def save_scan(self, tkinter_event=None) -> None:
        '''
        Method to save the data, you can add more logic later for other filetypes.
        The event input is to catch the tkinter event that is supplied but not used.
        '''
        allowed_formats = [('Image with dataset', '*.png'), ('Dataset', '*.hdf5')]

        # Default filename
        default_name = f'scan{self.id}_{self.timestamp.strftime("%Y%m%d")}'
            
        # Get the savefile name
        afile = tk.filedialog.asksaveasfilename(filetypes=allowed_formats, 
                                                initialfile=default_name+'.png',
                                                initialdir = self.parent_application.last_save_directory)
        # Handle if file was not chosen
        if afile is None or afile == '':
            logger.warning('File not saved!')
            return # selection was canceled.

        # Get the path
        file_path = '/'.join(afile.split('/')[:-1])  + '/'
        self.parent_application.last_save_directory = file_path # Save the last used file path
        logger.info(f'Saving files to directory: {file_path}')
        # Get the name with extension (will be overwritten)
        file_name = afile.split('/')[-1]
        # Get the filetype
        file_type = file_name.split('.')[-1]
        # Get the filename without extension
        file_name = '.'.join(file_name.split('.')[:-1]) # Gets everything but the extension

        # If the file type is .png, want to save image and hdf5
        if file_type == 'png':
            logger.info(f'Saving the PNG as {file_name}.png')
            fig = self.view.data_viewport.fig
            fig.savefig(file_path+file_name+'.png', dpi=300, bbox_inches=None, pad_inches=0)

        # Save as hdf5
        with h5py.File(file_path+file_name+'.hdf5', 'w') as df:
            
            logger.info(f'Saving the HDF5 as {file_name}.hdf5')
            
            # Save the file metadata
            ds = df.create_dataset('file_metadata', 
                                   data=np.array(['application', 
                                                  'qdlutils_version', 
                                                  'scan_id', 
                                                  'timestamp', 
                                                  'original_name'], dtype='S'))
            ds.attrs['application'] = 'qdlutils.qdlscan.LineScanApplication'
            ds.attrs['qdlutils_version'] = qdlutils.__version__
            ds.attrs['scan_id'] = self.id
            ds.attrs['timestamp'] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ds.attrs['original_name'] = file_name

            # Save the scan settings
            # If your implementation settings vary you should change the attrs
            ds = df.create_dataset('scan_settings/axis', data=self.axis)
            ds.attrs['units'] = 'None'
            ds.attrs['description'] = 'Axis of the scan.'
            ds = df.create_dataset('scan_settings/range', data=self.range)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Length of the scan.'
            ds = df.create_dataset('scan_settings/n_pixels', data=self.n_pixels)
            ds.attrs['units'] = 'None'
            ds.attrs['description'] = 'Number of pixels in the scan.'
            ds = df.create_dataset('scan_settings/time', data=self.time)
            ds.attrs['units'] = 'Seconds'
            ds.attrs['description'] = 'Length of time for the scan.'
            ds = df.create_dataset('scan_settings/time_per_pixel', data=self.time_per_pixel)
            ds.attrs['units'] = 'Seconds'
            ds.attrs['description'] = 'Time integrated per pixel.'

            ds = df.create_dataset('scan_settings/start_position_vector', data=self.start_position_vector)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Intial position of the scan.'
            ds = df.create_dataset('scan_settings/start_position_axis', data=self.start_position_axis)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Initial position on the scan axis.'
            ds = df.create_dataset('scan_settings/final_position_axis', data=self.final_position_axis)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Final position on the scan axis.'
            ds = df.create_dataset('scan_settings/min_position', data=self.min_position)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Minimum axis position of the scan.'
            ds = df.create_dataset('scan_settings/max_position', data=self.max_position)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Maximum axis position of the scan.'

            # Data
            ds = df.create_dataset('data/positions', data=self.data_x)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Positions of the scan (along axis).'
            ds = df.create_dataset('data/count_rates', data=self.data_y)
            ds.attrs['units'] = 'Counts per second'
            ds.attrs['description'] = 'Count rates measured over scan.'
            ds = df.create_dataset('data/counts', data=self.data_y*self.time_per_pixel)
            ds.attrs['units'] = 'Counts'
            ds.attrs['description'] = 'Counts measured over scan.'




class ImageScanApplication():
    '''
    This is the image scan application class for `qdlscan` which manages the data
    and GUI output for a single scan along a particular axis. It is meant to handle
    2-d confocal images, but could be used for scans in any pair of axes.
    '''

    def __init__(self,
                 parent_application: LauncherApplication,
                 application_controller: ScanController,
                 axis_1: str,
                 axis_2: str,
                 range: float,
                 n_pixels: int,
                 time: float,
                 id: str):
        
        
        self.parent_application = parent_application
        self.application_controller = application_controller
        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.range = range
        self.n_pixels = n_pixels
        self.time = time
        self.rclick_mpl_position_x = None
        self.rclick_mpl_position_y = None

        # Time per pixel
        self.time_per_pixel = time / n_pixels

        # Cmap for plotting
        self.cmap = DEFAULT_COLOR_MAP

        self.id = id
        self.timestamp = datetime.datetime.now()

        # Get the limits of the position for the axis
        if axis_1 == 'x':
            min_allowed_position_1 = self.parent_application.min_x_position
            max_allowed_position_1 = self.parent_application.max_x_position
        elif axis_1 == 'y':
            min_allowed_position_1 = self.parent_application.min_y_position
            max_allowed_position_1 = self.parent_application.max_y_position
        elif axis_1 == 'z':
            min_allowed_position_1 = self.parent_application.min_z_position
            max_allowed_position_1 = self.parent_application.max_z_position
        else:
            raise ValueError(f'Requested axis_1 {axis_1} is invalid.')
        if axis_2 == 'x':
            min_allowed_position_2 = self.parent_application.min_x_position
            max_allowed_position_2 = self.parent_application.max_x_position
        elif axis_2 == 'y':
            min_allowed_position_2 = self.parent_application.min_y_position
            max_allowed_position_2 = self.parent_application.max_y_position
        elif axis_2 == 'z':
            min_allowed_position_2 = self.parent_application.min_z_position
            max_allowed_position_2 = self.parent_application.max_z_position
        else:
            raise ValueError(f'Requested axis_2 {axis_2} is invalid.')

        # Get the starting position
        self.start_position_vector = application_controller.get_position()
        self.start_position_axis_1 = self.start_position_vector[AXIS_INDEX[axis_1]]
        self.start_position_axis_2 = self.start_position_vector[AXIS_INDEX[axis_2]]

        # Get the limits of the scan on axis 1
        self.min_position_1 = self.start_position_axis_1 - (range/2)
        self.max_position_1 = self.start_position_axis_1 + (range/2)
        # Check if the limits exceed the range and shift range to edge
        if self.min_position_1 < min_allowed_position_1:
            # Too close to minimum edge, need to shift up
            shift = min_allowed_position_1 - self.min_position_1
            self.min_position_1 += shift
            self.max_position_1 += shift
            self.start_position_axis_1 += shift
        if self.max_position_1 > max_allowed_position_1:
            # Too close to minimum edge, need to shift up
            shift = max_allowed_position_1 - self.max_position_1
            self.min_position_1 += shift
            self.max_position_1 += shift
            self.start_position_axis_1 += shift
        # Get the limits of the scan on axis 2
        self.min_position_2 = self.start_position_axis_2 - (range/2)
        self.max_position_2 = self.start_position_axis_2 + (range/2)
        # Check if the limits exceed the range and shift range to edge
        if self.min_position_2 < min_allowed_position_2:
            # Too close to minimum edge, need to shift up
            shift = min_allowed_position_2 - self.min_position_2
            self.min_position_2 += shift
            self.max_position_2 += shift
            self.start_position_axis_2 += shift
        if self.max_position_2 > max_allowed_position_2:
            # Too close to minimum edge, need to shift up
            shift = max_allowed_position_2 - self.max_position_2
            self.min_position_2 += shift
            self.max_position_2 += shift
            self.start_position_axis_2 += shift

        # Get the scan positions
        self.data_x = np.linspace(start=self.min_position_1, 
                                  stop=self.max_position_1, 
                                  num=n_pixels)
        self.data_y = np.linspace(start=self.min_position_2, 
                                  stop=self.max_position_2, 
                                  num=n_pixels)
        # To hold scan results
        self.data_z = np.empty(shape=(n_pixels, n_pixels))
        self.data_z[:,:] = np.nan

        # Launch the line scan GUI
        # Then initialize the GUI
        self.root = tk.Toplevel()
        self.root.title(f'Scan {id} ({self.timestamp.strftime("%Y-%m-%d %H:%M:%S")})')
        self.view = ImageScanApplicationView(window=self.root, 
                                            application=self,
                                            settings_dict=parent_application.scan_parameters)
        
        # Bind the buttons
        self.view.control_panel.pause_button.bind("<Button>", self.pause_scan)
        self.view.control_panel.continue_button.bind("<Button>", self.continue_scan)
        self.view.control_panel.save_button.bind("<Button>", self.save_scan)
        self.view.control_panel.norm_button.bind("<Button>", self.set_normalize)
        self.view.control_panel.autonorm_button.bind("<Button>", self.auto_normalize)

        # Setup the callback for rightclicks on the figure canvas
        self.view.data_viewport.canvas.mpl_connect('button_press_event', 
                                                   lambda event: self.open_rclick(event) if event.button == 3 else None)
        # Set the commands for the right click
        self.view.rclick_menu.add_command(label='Go to position', command=self.rclick_go_to) 
        self.view.rclick_menu.add_separator() 
        self.view.rclick_menu.add_command(label='Open counter', command=self.rclick_open_counter) 

        # Launch the thread
        self.scan_thread = Thread(target=self.start_scan_thread_function)
        self.scan_thread.start()

    def continue_scan(self, tkinter_event=None):
        # Don't do anything if busy
        if self.application_controller.busy:
            logger.error('Controller is busy; cannot continue scan.')
        if self.current_scan_index == self.n_pixels:
            logger.error('Scan already completed.')
            return None
        # Start the scan thread to continue
        self.scan_thread = Thread(target=self.continue_scan_thread_function)
        self.scan_thread.start()
    
    def pause_scan(self, tkinter_event=None):
        '''
        Tell the scanner to pause scanning
        '''
        if self.application_controller.stop_scan is True:
            logger.info('Already waiting to pause.')
            return None
        if self.current_scan_index == self.n_pixels:
            logger.error('Scan already completed.')
            return None

        if self.application_controller.busy:
            logger.info('Pausing the scan...')
            # Query the controller to stop
            self.application_controller.stop_scan = True

    def save_scan(self, tkinter_event=None):
        '''
        Method to save the data, you can add more logic later for other filetypes.
        The event input is to catch the tkinter event that is supplied but not used.
        '''
        allowed_formats = [('Image with dataset', '*.png'), ('Dataset', '*.hdf5')]

        # Default filename
        default_name = f'scan{self.id}_{self.timestamp.strftime("%Y%m%d")}'
            
        # Get the savefile name
        afile = tk.filedialog.asksaveasfilename(filetypes=allowed_formats, 
                                                initialfile=default_name+'.png',
                                                initialdir = self.parent_application.last_save_directory)
        # Handle if file was not chosen
        if afile is None or afile == '':
            logger.warning('File not saved!')
            return # selection was canceled.

        # Get the path
        file_path = '/'.join(afile.split('/')[:-1])  + '/'
        self.parent_application.last_save_directory = file_path # Save the last used file path
        logger.info(f'Saving files to directory: {file_path}')
        # Get the name with extension (will be overwritten)
        file_name = afile.split('/')[-1]
        # Get the filetype
        file_type = file_name.split('.')[-1]
        # Get the filename without extension
        file_name = '.'.join(file_name.split('.')[:-1]) # Gets everything but the extension

        # If the file type is .png, want to save image and hdf5
        if file_type == 'png':
            logger.info(f'Saving the PNG as {file_name}.png')
            fig = self.view.data_viewport.fig
            fig.savefig(file_path+file_name+'.png', dpi=300, bbox_inches=None, pad_inches=0)

        # Save as hdf5
        with h5py.File(file_path+file_name+'.hdf5', 'w') as df:
            
            logger.info(f'Saving the HDF5 as {file_name}.hdf5')
            
            # Save the file metadata
            ds = df.create_dataset('file_metadata', 
                                   data=np.array(['application', 
                                                  'qdlutils_version', 
                                                  'scan_id', 
                                                  'timestamp', 
                                                  'original_name'], dtype='S'))
            ds.attrs['application'] = 'qdlutils.qdlscan.ImageScanApplication'
            ds.attrs['qdlutils_version'] = qdlutils.__version__
            ds.attrs['scan_id'] = self.id
            ds.attrs['timestamp'] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            ds.attrs['original_name'] = file_name

            # Save the scan settings
            # If your implementation settings vary you should change the attrs
            ds = df.create_dataset('scan_settings/axis_1', data=self.axis_1)
            ds.attrs['units'] = 'None'
            ds.attrs['description'] = 'First axis of the scan (which is scanned quickly).'
            ds = df.create_dataset('scan_settings/axis_2', data=self.axis_2)
            ds.attrs['units'] = 'None'
            ds.attrs['description'] = 'Second axis of the scan (which is scanned slowly).'
            ds = df.create_dataset('scan_settings/range', data=self.range)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Length of the scan.'
            ds = df.create_dataset('scan_settings/n_pixels', data=self.n_pixels)
            ds.attrs['units'] = 'None'
            ds.attrs['description'] = 'Number of pixels in the scan.'
            ds = df.create_dataset('scan_settings/time', data=self.time)
            ds.attrs['units'] = 'Seconds'
            ds.attrs['description'] = 'Length of time for the scan along axis 1.'
            ds = df.create_dataset('scan_settings/time_per_pixel', data=self.time_per_pixel)
            ds.attrs['units'] = 'Seconds'
            ds.attrs['description'] = 'Time integrated per pixel.'

            ds = df.create_dataset('scan_settings/start_position_vector', data=self.start_position_vector)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Intial position of the scan.'
            ds = df.create_dataset('scan_settings/start_position_axis_1', data=self.start_position_axis_1)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Initial position on the scan axis 1.'
            ds = df.create_dataset('scan_settings/start_position_axis_2', data=self.start_position_axis_2)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Initial position on the scan axis 2.'
            ds = df.create_dataset('scan_settings/min_position_1', data=self.min_position_1)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Minimum axis 1 position of the scan.'
            ds = df.create_dataset('scan_settings/min_position_2', data=self.min_position_2)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Minimum axis 2 position of the scan.'
            ds = df.create_dataset('scan_settings/max_position_1', data=self.max_position_1)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Maximum axis 1 position of the scan.'
            ds = df.create_dataset('scan_settings/max_position_2', data=self.max_position_2)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Maximum axis 2 position of the scan.'

            # Data
            ds = df.create_dataset('data/positions_axis_1', data=self.data_x)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Positions of the scan (along axis 1).'
            ds = df.create_dataset('data/positions_axis_2', data=self.data_y)
            ds.attrs['units'] = 'Micrometers'
            ds.attrs['description'] = 'Positions of the scan (along axis 2).'
            ds = df.create_dataset('data/count_rates', data=self.data_z)
            ds.attrs['units'] = 'Counts per second'
            ds.attrs['description'] = 'Count rates measured over 2-d scan.'
            ds = df.create_dataset('data/counts', data=self.data_z*self.time_per_pixel)
            ds.attrs['units'] = 'Counts'
            ds.attrs['description'] = 'Counts measured over 2-d scan.'

    def start_scan_thread_function(self):
        '''
        This is the thread scan function for starting a scan
        '''
        try:
            self.current_scan_index = 0
            for line in self.application_controller.scan_image(
                            axis_1=self.axis_1,
                            start_1=self.min_position_1,
                            stop_1=self.max_position_1,
                            n_pixels_1=self.n_pixels,
                            axis_2=self.axis_2,
                            start_2=self.min_position_2,
                            stop_2=self.max_position_2,
                            n_pixels_2=self.n_pixels,
                            scan_time=self.time):
                # Set the data to the recently calculated line (in counts/second)
                self.data_z[self.current_scan_index] = line / self.time_per_pixel
                # Update the figure
                self.view.update_figure()
                # Increase the current scan index
                self.current_scan_index += 1

                logger.debug('Row complete.')

            self.home_position()
            # Update the figure
            self.view.update_figure()
            logger.info('Scan complete.')

        except Exception as e:
            logger.info(e)
        # Enable the buttons
        self.parent_application.enable_buttons()

    def continue_scan_thread_function(self):
        '''
        This is is the thread function to continue the scan from the middle.

        It would probably be the same as the start scan thread function if the 
        current_scan_index was set to begin with.
        '''
        try:
            for line in self.application_controller.scan_image(
                            axis_1=self.axis_1,
                            start_1=self.min_position_1,
                            stop_1=self.max_position_1,
                            n_pixels_1=self.n_pixels,
                            axis_2=self.axis_2,
                            start_2= self.data_y[self.current_scan_index], # Start at the index of the next queued scan
                            stop_2=self.max_position_2,
                            n_pixels_2=(self.n_pixels - self.current_scan_index), # Do the remaining pixels
                            scan_time=self.time):
                # Set the data to the recently calculated line (in counts/second)
                self.data_z[self.current_scan_index] = line / self.time_per_pixel
                # Update the figure
                self.view.update_figure()
                # Increase the current scan index
                self.current_scan_index += 1

                logger.debug('Row complete.')

            self.home_position()
            # Update the figure
            self.view.update_figure()
            logger.info('Scan complete.')

        except Exception as e:
            logger.info(e)
        # Enable the buttons
        self.parent_application.enable_buttons()

    def home_position(self):

        '''
        Go to the center of the scan
        '''
        # Move to optmial position
        self.application_controller.set_axis(axis=self.axis_1, position=self.start_position_axis_1)
        # Move to optmial position
        self.application_controller.set_axis(axis=self.axis_2, position=self.start_position_axis_2)

    def open_rclick(self, mpl_event : tk.Event = None):
        '''
        This function is the callback for the matplotlib right click event on the figure
        itself. It first gets and saves the x/y coordinate (in the data axes) and saves
        it to the appliction. It then gets the Tkinter position of the mouse and uses
        that to open a right click menu.
        '''
        # Get the right click position in the matplotlib axis
        self.rclick_mpl_position_x = mpl_event.xdata
        self.rclick_mpl_position_y = mpl_event.ydata
        # Get the tkinter x,y coordinates of the mouse
        mouse_x, mouse_y = self.root.winfo_pointerxy()
        # Open the popup menu with commands defined in the GUI file
        try: 
            self.view.rclick_menu.tk_popup(mouse_x,mouse_y) 
        finally: 
            self.view.rclick_menu.grab_release()
        
    def rclick_go_to(self):
        '''
        Method for the "go to" command in the right click menu. If a point within the 
        figure axis has been clicked, then it attempts to move to that position.
        '''
        # Make sure that the position of the right click is on the canvas
        if self.rclick_mpl_position_x and self.rclick_mpl_position_y:
            try:
                # Move to optmial position
                self.application_controller.set_axis(axis=self.axis_1, position=self.rclick_mpl_position_x)
                # Move to optmial position
                self.application_controller.set_axis(axis=self.axis_2, position=self.rclick_mpl_position_y)
                # Update the figure
                self.view.update_figure()
            except Exception as e:
                logger.warning(f'Error in moving to position: {e}')
        else:
            logger.warning('Right click position out of axis.')

    def rclick_open_counter(self):
        '''
        Method for the "open counter" command in the right click menu. Opens an instance
        of `qdlscope`.
        '''
        try:
            qdlscope.main(is_root_process=False)
        except Exception as e:
            logger.warning(f'{e}')

    def set_normalize(self, tkinter_event: tk.Event = None) -> None:
        '''
        Callback function to set the normalization of the figure based off of the values
        written to the GUI.
        '''
        # Get the value from the controller
        norm_min = float(self.view.control_panel.image_minimum.get())
        norm_max = float(self.view.control_panel.image_maximum.get())

        if norm_min > norm_max:
            raise ValueError(f'Minimum norm value {norm_min} > {norm_max}.')
        if norm_min < 0:
            raise ValueError(f'Minimum norm cannot be less than 0.')

        # Save the minimum/maximum norm
        self.view.norm_min = norm_min
        self.view.norm_max = norm_max

        self.view.update_figure()

    def auto_normalize(self, tkinter_event: tk.Event = None) -> None:
        '''
        Callback function to autonormalize the figure.
        '''
        # Save the minimum/maximum norm
        self.view.norm_min = None
        self.view.norm_max = None

        self.view.update_figure()


def main(is_root_process=True):
    tkapp = LauncherApplication(
        default_config_filename=DEFAULT_CONFIG_FILE,
        is_root_process=is_root_process)
    tkapp.run()

if __name__ == '__main__':
    main()
