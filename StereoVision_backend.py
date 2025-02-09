import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
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
        # Connect the Plot_signal signal after the stereo_vision object is created
        self.stereo_vision.Plot_signal.connect(self.display_plots)
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
        self.left_image = cv2.cvtColor(left_pic, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for display
        self.depth_map = depth_map

        # Convert the left image to a QImage and store it
        qimg1 = QImage(self.left_image.data, self.left_image.shape[1], self.left_image.shape[0], QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg1)

        # Set the image in the label
        self.ui.image_label.setPixmap(self.original_pixmap)

    def display_plots(self, img1_lines, img2_lines, disparity_rescaled, disparity_color, depth_rescaled, depth_color):
        """ Handle the plots in the main thread """
        try:
            # Display epipolar lines
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(img1_lines, cmap='gray')
            plt.title('Image 1 with Epipolar Lines'), plt.axis('off')
            plt.subplot(122), plt.imshow(img2_lines, cmap='gray')
            plt.title('Image 2 with Epipolar Lines'), plt.axis('off')
            plt.show()

            # Display disparity maps
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(disparity_rescaled, cmap='gray')
            plt.title('Disparity Map (Grayscale)'), plt.axis('off')
            plt.subplot(122), plt.imshow(disparity_color)
            plt.title('Disparity Map (Color)'), plt.axis('off')
            plt.show()

            # Display depth maps
            plt.figure(figsize=(10, 5))
            plt.subplot(121), plt.imshow(depth_rescaled, cmap='gray')
            plt.title('Depth Map (Grayscale)'), plt.axis('off')
            plt.subplot(122), plt.imshow(depth_color)
            plt.title('Depth Map (Color)'), plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Error while plotting: {e}")

    def set_cam0(self, text):
        self.K1 = text.strip('[]').split('; ')
        self.K1 = np.array([list(map(float, info.split())) for info in self.K1])  # Convert to NumPy array

    def set_cam1(self, text):
        self.K2 = text.strip('[]').split('; ')
        self.K2 = np.array([list(map(float, info.split())) for info in self.K2])  # Convert to NumPy array


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
        if self.left_image is None:  # Check if the image is loaded
            print("Error: Left image is not loaded.")
            return

        # Get mouse click position relative to the image label
        x = event.position().x()
        y = event.position().y()

        # Get QLabel dimensions
        label_width = self.ui.image_label.width()
        label_height = self.ui.image_label.height()

        # Get image dimensions using numpy array shape
        img_width = self.left_image.shape[1]  # Width of the image
        img_height = self.left_image.shape[0]  # Height of the image

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
     
    Plot_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)  # signal for plotting

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

    # Plot epipolar lines
    def plot_epipolar_lines(self, img1, img2, lines, pts1, pts2):
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
            img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
        return img1, img2
    
    def run(self):
        try:
            # Apply Gaussian blurring to reduce noise
            self.img1 = cv2.GaussianBlur(self.img1, (5, 5), 0)
            self.img2 = cv2.GaussianBlur(self.img2, (5, 5), 0)

            # Initialize SIFT detector and find keypoints and descriptors
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(self.img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(self.img2, None)

            # Feature matching using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched keypoints' coordinates
            pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            # Estimate Fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
            pts1_inliers = pts1[mask.ravel() == 1]
            pts2_inliers = pts2[mask.ravel() == 1]
            
            # Decompose the Essential matrix into rotation and translation
            E = self.K2.T @ F @ self.K1
            _, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, self.K1)

            # Rectify images
            retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_inliers, pts2_inliers, F,  (self.width, self.height))
            print("Homography for Image 1:\n", H1)
            print("Homography for Image 2:\n", H2)

            img1_rectified = cv2.warpPerspective(self.img1, H1, self.img1.shape[1::-1])
            img2_rectified = cv2.warpPerspective(self.img2, H2, self.img2.shape[1::-1])

            #This step calculates these lines to visualize the accuracy of the rectification and the corresponding points.
            lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
            lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

            img1_lines, img2_lines = self.plot_epipolar_lines(self.img1.copy(), self.img2.copy(), lines1, pts1_inliers, pts2_inliers)

           

            # StereoSGBM setup
            # Various parameters are tuned for better accuracy.
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disp,  # Keep this as a multiple of 16
                blockSize=7,  # reducing block size for better precision
                P1=8 * 3 * 7 ** 2,  # Adjust according to blockSize
                P2=32 * 3 * 7 ** 2,  # Adjust according to blockSize
                disp12MaxDiff=1,
                uniquenessRatio=5,  # Lower this to help refine matches
                speckleWindowSize=50,  # Decrease this to reduce speckle
                speckleRange=16,  # Reduce speckle range
                preFilterCap=31,  # Fine-tune this filter
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )


            # Compute the disparity map
            disparity = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0

            # Normalize and save the disparity map
            disparity_rescaled = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disparity_rescaled = np.uint8(disparity_rescaled)

            cv2.imwrite("disparity_grayscale.png", disparity_rescaled)

            # Convert the disparity map to a color image
            disparity_color = cv2.applyColorMap(disparity_rescaled, cv2.COLORMAP_JET)
            cv2.imwrite("disparity_color.png", disparity_color)

            

            # Calculate depth map
            focal_length = self.K1[0, 0]  # in pixels
            #this gives the distance from the camera to the object in the scene.
            depth = (focal_length * self.baseline) / (disparity_rescaled.astype(np.float32) + 1e-6)
            
            self.update_function(self.img1, depth)

            # Normalize and save depth images
            depth_rescaled = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_rescaled, cv2.COLORMAP_JET)

            cv2.imwrite("depth_gray.png", depth_rescaled)
            cv2.imwrite("depth_color.png", depth_color)

            

            self.Plot_signal.emit(img1_lines, img2_lines, disparity_rescaled, disparity_color, depth_rescaled, depth_color)

        except Exception as e:
            print(f"An error occurred: {e}")
