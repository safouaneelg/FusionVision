import pyrealsense2 as rs
import cv2
import numpy as np
import time
import os
import threading

# Declare should_save_image as a global variable
should_save_image = False
close_captured_image_window = False

def check_camera_connection():
    """Check if camera is connected"""
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        context = rs.context() 
        devices = context.query_devices()
        for device in devices:
            print("Device Information:")
            print("Device Product Line: ", str(device.get_info(rs.camera_info.product_line)))
            print("Device Serial Number: ", str(device.get_info(rs.camera_info.serial_number)))
        pipeline.stop()
    except:
        print("Camera not connected!")
        exit()

def start_realtime_stream():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    global should_save_image, close_captured_image_window

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(aligned_depth_frame.get_data())

            # Normalize depth image to range [0, 1] for better visualization
            depth_image = (depth_image / 10000.0 * 255.0).astype(np.uint8)

            # Apply a color map to the depth image
            depth_color_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            # Concatenate depth and color images horizontally
            #combined_image = np.concatenate((color_image, depth_color_image), axis=1)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)

            key = cv2.waitKey(1)
            if key == ord('c'):
                capture_image(color_image)
            elif key == ord('s'):
                should_save_image = True

            if close_captured_image_window:
                if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) <= 0:
                    close_captured_image_window = False
                else:
                    cv2.destroyWindow('Captured Image')

            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
def capture_image(image):
    cv2.namedWindow('Captured Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Captured Image', image)
    cv2.setWindowTitle('RealSense', 'Press \'s\' to save')

    global should_save_image, close_captured_image_window

    while True:
        key = cv2.waitKey(25) & 0xFF
        if key == ord('s'):
            should_save_image = True
            close_captured_image_window = True
        elif key == 27:
            break

        if close_captured_image_window:
            if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) <= 0:
                close_captured_image_window = False
            else:
                cv2.destroyWindow('Captured Image')
                break

    if should_save_image:
        save_image(image)

def save_image(image):
    global should_save_image
    if should_save_image:
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        image_name = 'dataset/' + str(int(time.time() * 1000)) + '.png'
        cv2.imwrite(image_name, image)
        print('Image saved at:', image_name)
        should_save_image = False

if __name__ == '__main__':
    check_camera_connection()
    start_realtime_stream()