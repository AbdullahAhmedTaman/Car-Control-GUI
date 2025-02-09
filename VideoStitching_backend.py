import sys
import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from VideoStitching_UI import Ui_Stitch_Dialog
import cv2
import numpy as np
import time

class VideoStitching(QDialog):
    def __init__(self):
        super().__init__()

        self.ui = Ui_Stitch_Dialog()
        self.ui.setupUi(self)

        # Variables for paths
        self.left_path = ''
        self.right_path = ''
        self.imageProcessing = None
        self.videoLooper_left = None
        self.videoLooper_right = None
        self.videoLooper_stitched = None

        # Connecting signals
        self.ui.leftOpen_pushButton.clicked.connect(self.getFileName_left)
        self.ui.rightOpen_pushButton.clicked.connect(self.getFileName_right)
        self.ui.start_pushButton.clicked.connect(self.start_videoStitching)
        self.ui.stop_pushButton.clicked.connect(self.stop_videoStitching)

    def getFileName_left(self):
        file_filter = 'Video File (*.mp4 *.avi *.mkv)'
        response_left = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Video File (*.mp4 *.avi *.mkv)'
        )
        self.left_path = response_left[0]  # Use instance variable

    def getFileName_right(self):
        file_filter = 'Video File (*.mp4 *.avi *.mkv)'
        response_right = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Video File (*.mp4 *.avi *.mkv)'
        )
        self.right_path = response_right[0]  # Use instance variable

    def start_videoStitching(self):
        if self.left_path and self.right_path:
            # Hide videos during processing
            self.ui.leftVideo_label.clear()
            self.ui.rightVideo_label.clear()
            self.ui.stitchedVideo_label.clear()

            # Reset the progress bar
            self.ui.progressBar.setValue(0)

            # Create and start the processing thread
            self.imageProcessing = ImageProcessing(self.left_path, self.right_path, self.ui.progressBar)
            self.imageProcessing.finished_stitching.connect(self.loop_videos)  # Connect only when stitching is done
            self.imageProcessing.start()

    def stop_videoStitching(self):
        # Stop all running threads
        if self.videoLooper_left:
            self.videoLooper_left.stop()
        if self.videoLooper_right:
            self.videoLooper_right.stop()
        if self.videoLooper_stitched:
            self.videoLooper_stitched.stop()

    def loop_videos(self, stitched_path):
        """
        Once the stitched video is finished, display the left, right, and stitched videos in separate loops.
        """
        if not stitched_path:
            return

        # Create individual threads for left, right, and stitched video loops
        self.videoLooper_left = VideoLooper(self.left_path, self.update_left_video)
        self.videoLooper_right = VideoLooper(self.right_path, self.update_right_video)
        self.videoLooper_stitched = VideoLooper(stitched_path, self.update_stitched_video)

        # Start each video thread
        self.videoLooper_left.start()
        self.videoLooper_right.start()
        self.videoLooper_stitched.start()

    def update_left_video(self, pic_left):
        self.ui.leftVideo_label.setPixmap(QPixmap.fromImage(pic_left))

    def update_right_video(self, pic_right):
        self.ui.rightVideo_label.setPixmap(QPixmap.fromImage(pic_right))

    def update_stitched_video(self, stitched_pic):
        self.ui.stitchedVideo_label.setPixmap(QPixmap.fromImage(stitched_pic))


class ImageProcessing(QThread):
    finished_stitching = pyqtSignal(str)  # Signal to indicate when stitching is done

    def __init__(self, left_path, right_path, progress_bar):
        super().__init__()
        self.left_path = left_path
        self.right_path = right_path
        self.progress_bar = progress_bar
        self.thread_active = False
        self.output_path = "stitched_output.mp4"  # Output stitched video path

    def run(self):
        self.thread_active = True
        desired_width = 640
        desired_height = 480
        desired_size = (desired_width, desired_height)
        fixed_size = (1280, 720)

        # Open the left and right video captures
        cap_left = cv2.VideoCapture(self.left_path)
        cap_right = cv2.VideoCapture(self.right_path)

        if not cap_left.isOpened() or not cap_right.isOpened():
            print("Error opening video streams")
            self.finished_stitching.emit(None)
            return

        # Get FPS and frame count
        fps = cap_left.get(cv2.CAP_PROP_FPS)
        total_frames = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT), cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

        # Initialize video writer for saving the stitched video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        out = cv2.VideoWriter(self.output_path, fourcc, fps, fixed_size)  # Create the VideoWriter object

        # Using the default stitching method
        image_stitcher = cv2.Stitcher.create()

        frame_count = 0  # Initialize the current frame count

        while self.thread_active:
            ret_left, frame_left = cap_left.read()
            ret_right = False
            frame_right = None
            if ret_left:
                ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                break

            # Resize both frames to the desired size before stitching
            frame_left_resized = cv2.resize(frame_left, desired_size)
            frame_right_resized = cv2.resize(frame_right, desired_size)

            # Perform stitching
            frames = [frame_left_resized, frame_right_resized]
            error, stitched_image = image_stitcher.stitch(frames)

            if error == cv2.Stitcher_OK:
                # Resize the stitched image to a fixed size and write to the video output
                stitched_image = cv2.resize(stitched_image, fixed_size)
                out.write(stitched_image)
            else:
                print(f"Stitching failed with error code: {error}")
                continue

            # Update progress bar (must be done in the main thread via signal)
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.ConnectionType.QueuedConnection, Q_ARG(int, progress))

            if frame_count == total_frames:
                break

        cap_left.release()
        cap_right.release()
        out.release()

        # Emit the output path once stitching is finished
        QMetaObject.invokeMethod(self.progress_bar, "setValue", Qt.ConnectionType.QueuedConnection, Q_ARG(int, 100))  # Set progress to 100 when done
        self.finished_stitching.emit(self.output_path)

    def stop(self):
        self.thread_active = False
        self.wait()


class VideoLooper(QThread):
    image_update = pyqtSignal(QImage)

    def __init__(self, video_path, update_function):
        super().__init__()
        self.video_path = video_path
        self.thread_active = False
        self.update_function = update_function

    def run(self):
        self.thread_active = True
        cap = cv2.VideoCapture(self.video_path)

        # Get the video's FPS (Frames Per Second)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:  # Default to 30 FPS if unable to get FPS from the video
            fps = 30.0

        # Calculate the time per frame in seconds
        frame_duration = 1 / fps

        while self.thread_active:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
                continue

            # Start measuring time
            start_time = time.time()

            # Convert frame to RGB and then to QImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimage = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)
            self.update_function(qimage.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))

            # Calculate the time taken to process and display the frame
            elapsed_time = time.time() - start_time

            # Calculate the remaining time to maintain the original frame rate
            delay_time = max(frame_duration - elapsed_time, 0)

            # Delay to keep playback speed in sync with the original frame rate
            time.sleep(delay_time)

        cap.release()

    def stop(self):
        self.thread_active = False
        self.wait()