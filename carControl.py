import serial
import time

# Initialize the serial connection (Make sure the port is correct)
arduino_port = "COM3"  # Update with your port name (e.g., COM3 for Windows, /dev/ttyUSB0 for Linux)
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

time.sleep(2)  # Wait for the Arduino to reset


def send_command(command):
    """Sends a command to the Arduino."""
    command=command+"\r"
    ser.write(command.encode())
    time.sleep(0.1)  # Short delay to ensure command is processed


def receive_data():
    """Receives and parses sensor data from the Arduino."""
    if ser.in_waiting > 0:
        data = ser.readline().decode('utf-8').strip("\r\n") #if not convenient convert to utf-8 using str method
        if data:
            # Data format: voltage:current:distance
            try:
                voltage, current, distance, state= data.split(':')
                print(f"Voltage: {voltage} V, Current: {current} A, Distance: {distance} cm, State={state}")
            except ValueError:
                print("Failed to parse sensor data")

def forward():
    send_command("F")
def backward():
    send_command("B")
def move_right():
    send_command("R")
def move_left():
    send_command("L")
def stop():
    send_command("S")
def autonomous():
    send_command("A")
def high_speed():
    send_command("H")
def medium_speed():
    send_command("M")
def low_speed():
    send_command("Q")

