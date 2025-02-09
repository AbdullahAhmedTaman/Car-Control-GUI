from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
import cv2
import os
from MainWindow_UI import Ui_MainWindow
from VideoStitching_backend import VideoStitching
from StereoVision_backend import StereoVision
from credits_backend import credits
import serial
import time

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing UI")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        print("UI setup complete")

        # Variable to check if Arduino is connected
        self.arduino_connected = False
        self.ser = None  # Initialize the ser attribute

        # Connecting signals
        self.ui.start_pushButton.pressed.connect(self.start_camera_feed)
        self.ui.stop_pushButton.clicked.connect(self.stop_camera_feed)
        self.ui.screenshot_pushButton.clicked.connect(self.screen_shot)
        self.ui.videoStitching_pushButton.clicked.connect(self.open_videoStitching)
        self.ui.StereoVision_pushButton.clicked.connect(self.open_StereoVision)
        self.ui.credits_pushButton.clicked.connect(self.open_Credits)
        self.ui.autonomus_mode_pushButton.clicked.connect(self.set_mode_auto)
        self.ui.manual_mode_pushButton.clicked.connect(self.set_mode_manual)
        self.ui.high_speed_pushButton.clicked.connect(self.set_speed_high)
        self.ui.medium_speed_pushButton.clicked.connect(self.set_speed_medium)
        self.ui.low_speed_pushButton.clicked.connect(self.set_speed_low)
        self.ui.forward_pushButton.pressed.connect(self.start_moving_forward)
        self.ui.forward_pushButton.released.connect(self.stop_moving_forward)
        self.ui.backward_pushButton.pressed.connect(self.start_moving_backward)
        self.ui.backward_pushButton.released.connect(self.stop_moving_backward)
        self.ui.right_pushButton.pressed.connect(self.start_moving_right)
        self.ui.right_pushButton.released.connect(self.stop_moving_right)
        self.ui.left_pushButton.pressed.connect(self.start_moving_left)
        self.ui.left_pushButton.released.connect(self.stop_moving_left)
        self.ui.Stop_car_pushButton.pressed.connect(self.stop)
        self.ui.keyboard_pushButton.clicked.connect(self.setup_keyboard)
        self.ui.GUI_pushButton.clicked.connect(self.control_by_GUI)

        # Keep track of dialog windows to avoid opening multiple instances
        self.video_stitching_window = None
        self.stereo_vision_window = None
        self.credits = None

        # Variable for checking the keyboard
        self.iskeyboard = False  # Set it false to make the GUI the default control

        # Arduino communication
        self.arduino_port = "COM4"  # Update with your port name
        self.baud_rate = 9600

        # Timer to check Arduino connection
        self.connection_check_timer = QTimer()
        self.connection_check_timer.timeout.connect(self.check_arduino_connection)
        self.connection_check_timer.start(1000)  # Check connection every second

        # Initialize Arduino connection
        self.initialize_arduino()

        # Timers for continuous movement
        self.forward_timer = QTimer()
        self.forward_timer.timeout.connect(self.forward)
        self.backward_timer = QTimer()
        self.backward_timer.timeout.connect(self.backward)
        self.right_timer = QTimer()
        self.right_timer.timeout.connect(self.right)
        self.left_timer = QTimer()
        self.left_timer.timeout.connect(self.left)

        # Install event filter
        self.installEventFilter(self)

    def check_arduino_connection(self):
        """Periodically check if Arduino is connected."""
        if self.ser and self.arduino_connected:
            try:
                self.ser.write(b'')  # Send an empty byte to check the connection
            except (serial.SerialException, OSError):
                self.handle_arduino_disconnection()
        elif not self.arduino_connected:
            self.initialize_arduino()  # Attempt reconnection

    def update_connection_status(self, connected):
        if connected:
            self.ui.connection_reading.setText("Connected")
            self.ui.connection_reading.setStyleSheet(
                "background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        else:
            self.ui.connection_reading.setText("Error")
            self.ui.connection_reading.setStyleSheet(
                "background-color:rgb(255,0,0);\ncolor:#090f13;\nborder-radius:15px;")

    def handle_arduino_disconnection(self):
        """Handles the disconnection of Arduino."""
        if self.arduino_connected:
            print("Arduino disconnected")
            self.arduino_connected = False
            self.update_connection_status(False)
            if self.ser:
                self.ser.close()
            if hasattr(self, 'arduino_worker'):
                self.arduino_worker.stop()

    def handle_arduino_reconnection(self):
        """Handles the reconnection of Arduino."""
        print("Arduino reconnected")
        self.arduino_connected = True
        self.update_connection_status(True)
        self.initialize_arduino()

    def send_command(self, command):
        """Sends a command to the Arduino."""
        if self.ser and self.arduino_connected:  # Ensure that the serial connection is established
            try:
                command = command + "\r"
                self.ser.write(command.encode())
                time.sleep(0.1)  # Short delay to ensure command is processed
            except serial.SerialException as e:
                print(f"Failed to send command: {e}")
                self.handle_arduino_disconnection()

    def initialize_arduino(self):
        try:
            # Attempt to open the serial port
            self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Allow some time for Arduino to reset

            # If successful, start the Arduino worker thread
            self.arduino_worker = ArduinoWorker(self.ser)
            self.arduino_worker.dataReceived.connect(self.handle_arduino_data)
            self.arduino_worker.start()  # Start the worker thread
            self.arduino_connected = True  # Set the flag to True if connection is successful
            print("Arduino worker started")
            self.update_connection_status(True)
        except serial.SerialException as e:
            print(f"Failed to access {self.arduino_port}: {e}")
            self.handle_arduino_disconnection()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.handle_arduino_disconnection() 


    def control_by_GUI(self):
        self.iskeyboard = False
        self.ui.controller_reading.setText("GUI")
        self.ui.controller_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.forward_pushButton.setEnabled(True)
        self.ui.backward_pushButton.setEnabled(True)
        self.ui.right_pushButton.setEnabled(True)
        self.ui.left_pushButton.setEnabled(True)
    
    def handle_arduino_data(self, data):
        """Handle data received from Arduino in the main thread."""
        if not self.arduino_connected:
            return  # Skip processing if Arduino is not connected
        try:
            voltage, current, distance, state = data.split(':')
            # Update GUI elements if needed
            self.ui.voltage_reading.setText(f"{voltage} V")
            self.ui.current_reading.setText(f"{current} A")
            self.ui.distance_reading.setText(f"{distance} cm")
            self.ui.direction_reading.setText(state)
        except ValueError:
            print("Failed to parse sensor data")

    def setup_keyboard(self):
        self.iskeyboard = True
        self.ui.controller_reading.setText("Keyboard")
        self.ui.controller_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.forward_pushButton.setDisabled(True)
        self.ui.backward_pushButton.setDisabled(True)
        self.ui.right_pushButton.setDisabled(True)
        self.ui.left_pushButton.setDisabled(True)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress and self.iskeyboard:
            if event.key() == Qt.Key.Key_W:
                self.start_moving_forward()
            elif event.key() == Qt.Key.Key_S:
                self.start_moving_backward()
            elif event.key() == Qt.Key.Key_D:
                self.start_moving_right()
            elif event.key() == Qt.Key.Key_A:
                self.start_moving_left()
        elif event.type() == QEvent.Type.KeyRelease and self.iskeyboard:
            if event.key() == Qt.Key.Key_W:
                self.stop_moving_forward()
            elif event.key() == Qt.Key.Key_S:
                self.stop_moving_backward()
            elif event.key() == Qt.Key.Key_D:
                self.stop_moving_right()
            elif event.key() == Qt.Key.Key_A:
                self.stop_moving_left()
        return super(Window, self).eventFilter(source, event)

    def start_moving_forward(self):
        self.forward_timer.start(100)  # Call forward() every 100ms

    def stop_moving_forward(self):
        self.forward_timer.stop()

    def forward(self):
        self.send_command("F")
        self.ui.direction_reading.setText("F")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_backward(self):
        self.backward_timer.start(100)
        

    def stop_moving_backward(self):
        self.backward_timer.stop()
        

    def backward(self):
        self.send_command("B")
        self.ui.direction_reading.setText("B")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_right(self):
        self.right_timer.start(100)
        

    def stop_moving_right(self):
        self.right_timer.stop()

    def right(self):
        self.send_command("R")
        self.ui.direction_reading.setText("R")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def start_moving_left(self):
        self.left_timer.start(100)
        

    def stop_moving_left(self):
        self.left_timer.stop()

    def left(self):
        self.send_command("L")
        self.ui.direction_reading.setText("L")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")

    def stop(self):
        self.send_command("S")
        self.ui.direction_reading.setText("S")
        self.ui.direction_reading.setStyleSheet("background-color:rgb(255,0,0);\ncolor:#090f13;\nborder-radius:15px;")   

    def set_mode_auto(self):
        self.send_command("A")
        self.ui.mode_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("A")

    def set_mode_manual(self):
        self.send_command("S")
        self.ui.mode_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.mode_reading.setText("M")

    def set_speed_high(self):
        self.send_command("H")
        self.ui.speed_reading.setStyleSheet("background-color:rgb(0,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("H")
    
    def set_speed_medium(self):
        self.send_command("M")
        self.ui.speed_reading.setStyleSheet("background-color:rgb(255,255,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("M")

    def set_speed_low(self):
        self.send_command("L")
        self.ui.speed_reading.setStyleSheet("background-color:rgb(255,0,0);\ncolor:#090f13;\nborder-radius:15px;")
        self.ui.speed_reading.setText("L")    

    def start_camera_feed(self):
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.update_feed)

    def update_feed(self, Image):
        self.ui.camera_label.setPixmap(QPixmap.fromImage(Image))

    def stop_camera_feed(self):
        if hasattr(self, 'Worker1'):
            self.Worker1.stop()  # Gracefully stop the thread
            self.Worker1.quit()  # Ensure it quits after stopping
            self.Worker1.wait()  # Wait for the thread to finish
            self.Worker1.ImageUpdate.disconnect()  # Disconnect signal after stopping
            self.clear_feed()  # Clear the camera label


    def clear_feed(self):
        self.ui.camera_label.clear()

    def screen_shot(self):
        if hasattr(self, 'Worker1'):
            self.Worker1.take_screenshot()  # Trigger screenshot in the worker thread

    def open_videoStitching(self):
        # Check if the dialog is already open
        if self.video_stitching_window is None:
            # Instantiate the video stitching dialog
            self.video_stitching_window = VideoStitching()

        if not self.video_stitching_window.isVisible():
            self.video_stitching_window.show()
        else:
            self.video_stitching_window.raise_()  # Bring the dialog to the front

    def open_StereoVision(self):
        # Check if the dialog is already open
        if self.stereo_vision_window is None:
            # Instantiate the Stereo Vision dialog
            self.stereo_vision_window = StereoVision()

        if not self.stereo_vision_window.isVisible():
            self.stereo_vision_window.show()
        else:
            self.stereo_vision_window.raise_()  # Bring the dialog to the front

    def open_Credits(self):
        # Check if the dialog is already open
        if self.credits is None:
            # Instantiate the Credits dialog
            self.credits = credits()

        if not self.credits.isVisible():
            self.credits.show()
        else:
            self.credits.raise_()  # Bring the dialog to the front

    def closeEvent(self, event):
        # Stop the Arduino worker thread when the main window is closed
        if self.arduino_connected:
            self.arduino_worker.stop()
            self.arduino_worker.wait()

        # Clean up dialog windows
        if self.video_stitching_window:
            self.video_stitching_window.close()
        if self.stereo_vision_window:
            self.stereo_vision_window.close()
        if self.credits:
            self.credits.close()
        event.accept()

class ArduinoWorker(QThread):
    dataReceived = pyqtSignal(str)

    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self._is_running = True

    def run(self):
        while self._is_running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode('utf-8').strip("\r\n")
                    if data:
                        self.dataReceived.emit(data)
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                break
            except Exception as e:
                print(f"Unexpected error in Arduino worker: {e}")
                break

    def stop(self):
        self._is_running = False
        self.wait()



class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.ThreadActive = True
        self.screenshot_requested = False  # Flag for screenshot
        self.capture = None  # Initialize the video capture object

    def run(self):
        self.ThreadActive = True
        self.capture = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = self.capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

                # Emit the image for the live feed
                self.ImageUpdate.emit(Pic)

                # Save screenshot if requested
                if self.screenshot_requested:
                    self.save_screenshot(frame)
                    self.screenshot_requested = False

        # Release resources after the loop ends
        self.capture.release()

    def take_screenshot(self):
        """Method to request a screenshot."""
        self.screenshot_requested = True

    def save_screenshot(self, frame):
        """Method to save the screenshot."""
        screenshot_dir = 'screenshots'
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        filename = os.path.join(screenshot_dir, 'screenshot.png')
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    def stop(self):
        """Stop the thread and ensure a clean shutdown."""
        self.ThreadActive = False
        self.wait()  # Wait for the thread to properly exit before quitting



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
