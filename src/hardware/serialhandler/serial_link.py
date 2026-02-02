import serial
import time
import threading
import logging

class SerialLink:
    """
    Robust Serial Communication Class for RAVEN (Team C).
    
    Responsibilities:
    1. Manage USB Serial connection (open, close, reconnect).
    2. Maintain a consistent 10Hz transmission loop.
    3. Package speed/steer commands into the correct format.
    """
    
    def __init__(self, port='/dev/ttyACM0', baud=115200, frequency=10.0):
        self.port = port
        self.baud = baud
        self.interval = 1.0 / frequency
        self.serial_conn = None
        self.running = False
        
        # State variables to send
        self.target_speed = 0.0
        self.target_steer = 0.0
        
        # Threading
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._transmission_loop)
        self.thread.daemon = True
        
        self.logger = logging.getLogger("SerialLink")

    def start(self):
        """Starts the transmission loop."""
        self.running = True
        self.thread.start()
        self.logger.info("SerialLink started.")

    def stop(self):
        """Stops the loop and closes connection."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.disconnect()
        self.logger.info("SerialLink stopped.")

    def connect(self):
        """Attempts to open the serial port."""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                return True
            
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=1)
            self.logger.info(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            self.logger.warning(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Closes the serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            self.logger.info("Disconnected.")

    def set_drive_command(self, speed, steer):
        """Updates the desired speed and steer values."""
        with self.lock:
            self.target_speed = speed
            self.target_steer = steer

    def _transmission_loop(self):
        """Main loop sending data at fixed frequency."""
        while self.running:
            start_time = time.time()
            
            # 1. Ensure Connection
            if not self.serial_conn or not self.serial_conn.is_open:
                if not self.connect():
                    time.sleep(1) # Wait before retry
                    continue

            # 2. Package Data
            # Protocol: #SPEED:val;STEER:val;;
            # Note: Adjust formatting to match your specific firmware parser
            # We use the generic format described in the task
            with self.lock:
                spd = self.target_speed
                str_ang = self.target_steer
            
            # Using the format usually expected by the RAVEN Parser
            # You might need to adjust this to match MessageConverter if strictly required
            # But the task asked for a "Python Serial Class" to package strings.
            command_str = f"#SPEED:{spd:.2f};STEER:{str_ang:.2f};;\r\n"
            
            # 3. Send Data
            try:
                self.serial_conn.write(command_str.encode('ascii'))
            except serial.SerialException as e:
                self.logger.error(f"Write error: {e}")
                self.disconnect() # Force reconnect next loop

            # 4. Maintain Frequency (Sleep for remainder of interval)
            elapsed = time.time() - start_time
            sleep_time = self.interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == '__main__':
    # Local Test
    logging.basicConfig(level=logging.INFO)
    link = SerialLink(port='/dev/ttyACM0', frequency=10) # Adjust port for Mac if needed
    link.start()
    
    try:
        # Simulate some commands
        for i in range(50):
            link.set_drive_command(speed=i, steer=0.0)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()
