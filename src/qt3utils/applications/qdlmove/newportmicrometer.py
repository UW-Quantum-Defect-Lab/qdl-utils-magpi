import serial
import time


class NewportMicrometer():
    '''
    Controller for Newport automated micrometers via serial port
    '''
    def __init__(self,
                 port: str='COM4',
                 min: float=0.0,
                 max: float=25000.0,
                 timeout: float=10):

        self.port = port
        self.min = min
        self.max = max
        self.timeout = timeout

        self.ser = serial.Serial(
            port=port,
            baudrate=921600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            xonxoff=True
        )

        #Initialization
        #Emter configuration state
        self.ser.write('1PW1\r\n'.encode('utf-8'))
        #HT Set Home Search type
        self.ser.write('1HT\r\n'.encode('utf-8'))
        #BA Set backlash compensation
        self.ser.write('1BA0.003\r\n'.encode('utf-8'))
        #Set friction compensation
        self.ser.write('1FF05\r\n'.encode('utf-8'))
        #Leave configuration state
        self.ser.write('1PW0\r\n'.encode('utf-8'))

        #Execute Home Search, needed before controller can move
        self.ser.write('1OR\r\n'.encode('utf-8'))

    def go_to_position(self, position: float) -> None:
        
        # Check if the requested position is valid
        if self.is_valid_position(position):
            # Encoding needs to be in units of mm
            # Input is provided in units of microns so divide by 1e3
            # Generate the command and write to serial.
            command='1SE'+str(position/1000)+'\r\n'
            self.ser.write(command.encode('utf-8'))
            self.ser.write('SE\r\n'.encode('utf-8'))

            # Response of the micrometers is finite
            # This loop waits for the micrometers to finish movement.
            timeout_clock=0
            moving=True
            while moving and (timeout_clock < self.timeout):
                # Wait for 0.1 seconds
                time.sleep(0.1)
                # Check if position is within 0.1 microns of target
                if abs(self.read_position()-position)<0.1:
                    # Break from loop
                    moving=False
                # Increment the timeout clock
                timeout_clock+=0.1
        else:
            # Raise value error if the requested position is invalid.
            error_message = f'Requested position {position} is out of bounds.'
            raise ValueError(error_message)


    def read_position(self) -> float:
        '''
        Read the position of the micrometer
        '''
        # Get the read command and write to serial
        command='1TP\r\n'
        self.ser.write(command.encode('utf-8'))

        # Read the result and cast to float
        # The first 3 characters of the string are discarded
        raw=self.ser.readline()
        return float(raw[3:12]) * 1000
    

    def close(self) -> None:
        '''
        Closes the serial connection to micrometer
        '''
        # Send the close command to serial
        self.ser.write('SE\r\n'.encode('utf-8'))
        self.ser.close()


    def is_valid_position(self, value):
        '''
        Validates if value is within the allowed range
        '''
        return (value >= self.min) and (value <= self.max)
    