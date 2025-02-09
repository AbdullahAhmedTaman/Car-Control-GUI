import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import cv2
import numpy as np
from StereoVision_UI import Ui_StereoVision

class StereoVision(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_StereoVision()
        self.ui.setupUi(self)

        # Variables for camera parameters and stereo vision setup
        self.K1 = None
        self.K2 = None
        self.baseline = 0
        self.Width = 0
        self.Height = 0
        self.num_disp = None

        self.stereo_vision = None
        self.left_image = None
        self.right_image = None
        self.depth_map = None
        self.point1 = None
        self.point2 = None

        # Keep the original image without drawn points for refreshing
        self.original_pixmap = None

        # Variables for paths
        self.left_path = ''
        self.right_path = ''

        # Connecting signals
        self.ui.leftOpen_stereo_pushButton.clicked.connect(self.getFileName_left)
        self.ui.rightOpen_stereo_pushButton.clicked.connect(self.getFileName_right)
        self.ui.start_stereo_pushButton.clicked.connect(self.start_stereo_vision)
        self.ui.cam0.textChanged.connect(self.set_cam0)
        self.ui.cam1.textChanged.connect(self.set_cam1)
        self.ui.baseline.textChanged.connect(self.set_baseline)
        self.ui.height.textChanged.connect(self.set_height)
        self.ui.width.textChanged.connect(self.set_width)
        self.ui.num_disp.textChanged.connect(self.set_num_disp)

        # Set up mouse events
        self.ui.image_label.mousePressEvent = self.get_mouse_position

    def start_stereo_vision(self):
        # Set frame and label sizes based on user input (Width, Height)
        self.update_frame_and_label_size()

        self.stereo_vision = StereoVision_processing(
            self.left_path, self.right_path, self.K1, self.K2, self.baseline,
            self.Width, self.Height, self.num_disp, self.update_images)
        self.stereo_vision.start()

    def update_frame_and_label_size(self):
        # Resize the photo_back_frame and image_label based on user input (Height and Width)
        frame_height = self.Height + 20
        frame_width = self.Width + 20

        # Resize the photo_back_frame
        self.ui.photo_back_frame.setFixedSize(frame_width, frame_height)

        # Resize the image_label to exactly the size of user input (Height and Width)
        self.ui.image_label.setFixedSize(self.Width, self.Height)

    def update_images(self, left_pic, depth_map):
        self.left_image = left_pic
        self.depth_map = depth_map

        # Store the original pixmap to allow refreshing
        self.original_pixmap = QPixmap.fromImage(left_pic)

        # Set the image in the label
        self.ui.image_label.setPixmap(self.original_pixmap)

    def set_cam0(self, text):
        self.K1 = text.strip('[]').split('; ')
        self.K1 = [list(map(float, info.split())) for info in self.K1]

    def set_cam1(self, text):
        self.K2 = text.strip('[]').split('; ')
        self.K2 = [list(map(float, info.split())) for info in self.K2]

    def set_baseline(self, text):
        self.baseline = float(text)

    def set_height(self, text):
        self.Height = int(text)  # Ensure the height is an integer

    def set_width(self, text):
        self.Width = int(text)  # Ensure the width is an integer

    def set_num_disp(self, text):
        self.num_disp = int(text)

    def getFileName_left(self):
        file_filter = 'Image File (*.jpg *.png *.jpeg)'
        response_left = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Image File (*.jpg *.png *.jpeg)'
        )
        self.left_path = response_left[0]

    def getFileName_right(self):
        file_filter = 'Image File (*.jpg *.png *.jpeg)'
        response_right = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Image File (*.jpg *.png *.jpeg)'
        )
        self.right_path = response_right[0]

    def get_mouse_position(self, event):
        # Get mouse click position relative to the image label
        x = event.position().x()
        y = event.position().y()

        # Get QLabel dimensions
        label_width = self.ui.image_label.width()
        label_height = self.ui.image_label.height()

        # Get image dimensions
        img_width = self.left_image.width()
        img_height = self.left_image.height()

        # Calculate the scale factor used for the QLabel's image display
        pixmap = self.ui.image_label.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Handle aspect ratio scaling of the displayed image
        scale_x = img_width / pixmap_width
        scale_y = img_height / pixmap_height

        # Account for any empty margins added during aspect ratio scaling
        margin_x = (label_width - pixmap_width) // 2
        margin_y = (label_height - pixmap_height) // 2

        # Adjust x and y to the image's coordinates (scaling from QLabel to original image)
        img_x = int((x - margin_x) * scale_x)
        img_y = int((y - margin_y) * scale_y)

        # Ensure that the point is within the image bounds
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))

        # Fetch the Z (depth) value from the depth map
        z = self.depth_map[img_y, img_x]

        # Check if this is the first or second point
        if self.point1 is None:
            self.point1 = (img_x, img_y, z)
            self.redraw_points()
            self.ui.point1.setText(f"({img_x}, {img_y})")
        elif self.point2 is None:
            self.point2 = (img_x, img_y, z)
            self.redraw_points()
            self.calculate_distance()  # Calculate distance once the second point is set

    def redraw_points(self):
        # Reset the image to the original
        pixmap = self.original_pixmap.copy()
        painter = QPainter(pixmap)

        # Draw point 1 (if it exists) in red
        if self.point1:
            painter.setPen(QPen(Qt.GlobalColor.red, 8))
            img_x, img_y, _ = self.point1
            painter.drawPoint(img_x, img_y)

        # Draw point 2 (if it exists) in blue
        if self.point2:
            painter.setPen(QPen(Qt.GlobalColor.blue, 8))
            img_x, img_y, _ = self.point2
            painter.drawPoint(img_x, img_y)

        painter.end()  # Ensure QPainter is properly closed

        # Update the QLabel with the modified pixmap
        self.ui.image_label.setPixmap(pixmap)

        # Display the coordinates in the respective QLineEdit
        if self.point1:
            img_x, img_y, z = self.point1
            self.ui.point1.setText(f"({img_x}, {img_y})")

        if self.point2:
            img_x, img_y, z = self.point2
            self.ui.point2.setText(f"({img_x}, {img_y})")

    def calculate_distance(self):
        if self.point1 and self.point2:
            # Unpack x, y, and z for both points
            x1, y1, z1 = self.point1
            x2, y2, z2 = self.point2

            # Calculate pixel distance in 2D (ignoring depth for pixel distance)
            pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Display pixel distance
            self.ui.lenght.setText(f"{pixel_distance:.2f} pixels")

            # Reset points for the next selection
            self.point1 = None
            self.point2 = None



class StereoVision_processing(QThread):
    Image_update = pyqtSignal(QImage, np.ndarray)

    def __init__(self, left_path, right_path, K1, K2, baseline, width, height, num_disp, update_function):
        super().__init__()
        self.img1 = cv2.imread(left_path)  # Load image in color (RGB)
        self.img2 = cv2.imread(right_path)  # Load image in color (RGB)

        self.update_function = update_function
        self.K1 = K1
        self.K2 = K2
        self.baseline = baseline
        self.width = width
        self.height = height
        self.num_disp = num_disp

    def run(self):
        try:
            # Convert images from BGR (OpenCV default) to RGB
            img1_rgb = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)

            

            # Convert the left image to QImage for display
            qimg1 = QImage(img1_rgb.data, img1_rgb.shape[1], img1_rgb.shape[0], QImage.Format.Format_RGB888)

            # Perform stereo matching and depth map computation
            stereo = cv2.StereoSGBM_create(numDisparities=self.num_disp, blockSize=5)
            disparity = stereo.compute(cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY),
                                       cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0
            depth_map = self.K1[0][0] * self.baseline / (disparity + 1e-5)

            # Send the left image (QImage) and depth map to the main window for display
            self.update_function(qimg1, depth_map)

        except Exception as e:
            print(f"An error occurred: {e}")
