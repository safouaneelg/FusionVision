import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def get_rgbd(color_frame, depth_frame, align, depth_scale=200, depth_trunc=4, convert_rgb_to_intensity=False):
    depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.float32)

    color_image = np.asanyarray(color_frame.get_data())

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image),
        o3d.geometry.Image(depth_image),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=convert_rgb_to_intensity
    )

    return rgbd

def stream_realSense_with_depth_and_pointcloud():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window("RGB and Depth with 3D Point Cloud", width=1000, height=700)

    # Transformation matrix for flipping the point cloud upside down and left to right
    flip_matrix = np.array([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    points = o3d.geometry.PointCloud()
    visualizer.add_geometry(points)

    first = True

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            align = rs.align(rs.stream.color)
            aligned_frames = align.process(frames)

            rgbd = get_rgbd(aligned_frames.get_color_frame(), aligned_frames.get_depth_frame(), align)

            new_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                intrinsic=intrinsics,
            )

            new_pointcloud.transform(flip_matrix)
            
            points.points = new_pointcloud.points
            points.colors = new_pointcloud.colors

            color_image_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.2), cv2.COLORMAP_JET)
            color_image_gray_3d = cv2.cvtColor(color_image_gray, cv2.COLOR_RGB2BGR)

            # Concatenate color and depth images horizontally
            image_frame = np.hstack((color_image_gray_3d, depth_colormap))
            image_frame = image_frame.astype(np.uint8)

            cv2.imshow("Color and Depth Frames", image_frame)

            # Reset viewpoint in the first frame to look at the scene correctly
            if first:
                visualizer.reset_view_point(True)
                first = False

            visualizer.update_geometry(points)
            visualizer.poll_events()
            visualizer.update_renderer()

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        visualizer.destroy_window()

stream_realSense_with_depth_and_pointcloud()
