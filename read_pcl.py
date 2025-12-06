# Read camera params and image, then reconstruct the point cloud

import json
import os

import config as cfg

if cfg.USE_X11:
    os.environ["XDG_SESSION_TYPE"] = "x11"


import cv2
import numpy as np
import open3d as o3d

ALL_CAMERAS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
               "Boutput0", "Boutput1", "Boutput2", "Boutput3",
               "output0", "hololens"]

with open(os.path.join(cfg.DATA_PATH, "meta_data", "camera_matrix.json"), "r") as f:
    CAMERA_PARAMS = json.load(f)


def read_img_pair(camera_name: str, rgb_frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the closest depth frame by checking the timestamps, 
    and returns the (rgb, depth) pair.
    """
    with open(os.path.join(cfg.DATA_PATH, "meta_data", "timestamps.txt"), "r") as f:
        rgb_timestamps = f.readlines()
    rgb_timestamp = float(rgb_timestamps[rgb_frame_idx].strip())

    # find the index of closest timestamp in depth timestamps
    with open(os.path.join(cfg.DATA_PATH, "timestamps_depth", f"{camera_name}.txt"), "r") as f:
        depth_timestamps = f.readlines()
    depth_timestamps = [float(ts) for ts in depth_timestamps]

    depth_frame_idx = np.argmin(np.abs(depth_timestamps - rgb_timestamp))

    try:
        rgb_video = cv2.VideoCapture(os.path.join(cfg.DATA_PATH, "videos", f"{camera_name}.mp4"))
        depth_video = cv2.VideoCapture(os.path.join(cfg.DATA_PATH, "videos_depth", f"{camera_name}.mp4"))
        rgb_video.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_idx)
        depth_video.set(cv2.CAP_PROP_POS_FRAMES, depth_frame_idx)

        rgb_ret, rgb_frame = rgb_video.read()
        depth_ret, depth_frame = depth_video.read()

        if not rgb_ret:
            raise ValueError(f"Failed to read frame {rgb_frame_idx} from {camera_name}")
        if not depth_ret:
            raise ValueError(f"Failed to read frame {depth_frame_idx} from {camera_name}")

    finally:
        if rgb_video.isOpened():
            rgb_video.release()
        if depth_video.isOpened():
            depth_video.release()

    return (rgb_frame, depth_frame)

def read_img_pair_full_depth(camera_name: str, rgb_frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds the closest depth frame by checking the timestamps, 
    and returns the (rgb, depth) pair.
    """
    with open(os.path.join(cfg.DATA_PATH, "meta_data", "timestamps.txt"), "r") as f:
        rgb_timestamps = f.readlines()
    rgb_timestamp = float(rgb_timestamps[rgb_frame_idx].strip())

    # find the index of closest timestamp in depth timestamps
    with open(os.path.join(cfg.DATA_PATH, "timestamps_depth", f"{camera_name}.txt"), "r") as f:
        depth_timestamps = f.readlines()
    depth_timestamps = np.array([float(ts) for ts in depth_timestamps])

    depth_frame_idx = np.argmin(np.abs(depth_timestamps - rgb_timestamp)).item()

    try:
        rgb_video = cv2.VideoCapture(os.path.join(cfg.DATA_PATH, "videos", f"{camera_name}.mp4"))
        # depth_video = cv2.VideoCapture(os.path.join(data_path, "videos_depth", f"{camera_name}.mp4"))
        rgb_video.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_idx)
        # depth_video.set(cv2.CAP_PROP_POS_FRAMES, depth_frame_idx)

        rgb_ret, rgb_frame = rgb_video.read()
        # depth_ret, depth_frame = depth_video.read()

        if not rgb_ret:
            raise ValueError(f"Failed to read frame {rgb_frame_idx} from {camera_name}")
        # if not depth_ret:
        #     raise ValueError(f"Failed to read frame {depth_frame_idx} from {camera_name}")

    finally:
        if rgb_video.isOpened():
            rgb_video.release()
        # if depth_video.isOpened():
        #     depth_video.release()

    depth_frame = cv2.imread(os.path.join(cfg.FULL_DEPTH_DATA_PATH, "depth_images", f"{camera_name}", f"depth_{depth_frame_idx:08d}.png"), cv2.IMREAD_UNCHANGED)

    return (rgb_frame, depth_frame)



def create_pcl(rgb_frame: np.ndarray, depth_frame: np.ndarray, camera_params: dict) -> o3d.geometry.PointCloud:
    """
    Creates a point cloud from the rgb and depth frames.
    """
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frame),
            o3d.geometry.Image(depth_frame),
            depth_scale=1000.0,
            depth_trunc=5,
            convert_rgb_to_intensity=False,
        ),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=rgb_frame.shape[1], height=rgb_frame.shape[0], fx=camera_params["fx"], fy=camera_params["fy"], cx=camera_params["cx"], cy=camera_params["cy"]
        ),
        extrinsic=np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )
    return pcd

if __name__ == "__main__":
    rgb_frame = cv2.imread("samples/undistorted_rgb.png")
    depth_frame = cv2.imread("samples/undistorted_depth.png", cv2.IMREAD_UNCHANGED)
    depth_frame = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
    cv2.imwrite("samples/undistorted_depth_resized.png", depth_frame)

    K = CAMERA_PARAMS["Aoutput0"]["K"]
    params = {"fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2]}
    pcd = create_pcl(rgb_frame, depth_frame, params)
    o3d.visualization.draw_geometries([pcd])