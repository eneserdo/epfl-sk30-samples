# Read camera params and image, then reconstruct the point cloud

import json
import os

import pyimof  # just for optical flow

import config as cfg

if cfg.USE_X11:
    # Benim makinam için (wsl için gerekli)
    os.environ["XDG_SESSION_TYPE"] = "x11"


import cv2
import numpy as np
import open3d as o3d

ALL_CAMERAS = ["Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3",
               "Boutput0", "Boutput1", "Boutput2", "Boutput3",
               "output0", "hololens"]

with open(os.path.join("full_depth", "camera_matrix.json"), "r") as f:
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

    depth_frame_idx += 1 # because the depth frame index is 0-indexed in the depth video, but 1-indexed in the depth image files
    depth_frame = cv2.imread(os.path.join(cfg.FULL_DEPTH_DATA_PATH, "depth_images", f"{camera_name}", f"depth_{depth_frame_idx:08d}.png"), cv2.IMREAD_UNCHANGED)

    return (rgb_frame, depth_frame)


def viz_depth(depth: np.ndarray, depth_min: float = None, depth_max: float = None) -> np.ndarray:
    depth = depth.copy()

    # Remove outliers
    depth_min = np.percentile(depth, 5) if depth_min is None else depth_min
    depth_max = np.percentile(depth, 95) if depth_max is None else depth_max
    depth = np.clip(depth, depth_min, depth_max)

    # Normalize to 0-255
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)
    depth = cv2.putText(depth, f"Depth Min: {depth_min}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    depth = cv2.putText(depth, f"Depth Max: {depth_max}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return depth


def undistort(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Undistorts the image using the camera matrix and distortion coefficients.
    Returns the undistorted image and the new camera matrix.
    """
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
    dst = cv2.undistort(image, K, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return (dst, newcameramtx)


def viz_dist_flow(image: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Undistort image then calculate optical flow.
    Visualizes the distortion flow of the image.
    """
    # prefer duplication over the wrong abstraction 

    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
    undistorted = cv2.undistort(image, K, dist, None, newcameramtx)
    
    hsv = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    u, v = pyimof.solvers.tvl1(gray, undistorted_gray)
    mag, ang = cv2.cartToPolar(u, v)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    optical_flow_vis = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return optical_flow_vis


def align_color_to_depth(depth, color, camera_params: dict):
    Kc, dist_c = np.array(camera_params["K"]), np.array(camera_params["dist"])
    Kd, dist_d = np.array(camera_params["depth_K"]), np.array(camera_params["depth_dist"])
    world2cam, world2depth = np.array(camera_params["word2cam"]), np.array(camera_params["world2depth"])
    
    # Undistort images
    # depth, Kd = undistort(depth, Kd, dist_d)
    # color, Kc = undistort(color, Kc, dist_c)

    h, w = depth.shape[:2]
    color_out = np.zeros((h, w, 3), dtype=np.uint8)

    fx_d, fy_d = Kd[0,0], Kd[1,1]
    cx_d, cy_d = Kd[0,2], Kd[1,2]

    fx_c, fy_c = Kc[0,0], Kc[1,1]
    cx_c, cy_c = Kc[0,2], Kc[1,2]

    Rc = world2cam[:3, :3]
    tc = world2cam[:3, 3].reshape(3, 1)
    Rd = world2depth[:3, :3]
    td = world2depth[:3, 3].reshape(3, 1)

    """
    Given:
    Oc = Rc*Ow + tc
    Od = Rd*Ow + td
    Find:
    Oc in terms of Od
    =>
    Rc^(-1)*(Oc - tc) = Ow
    Rd^(-1)*(Od - td) = Ow
    Rc^(-1)*(Oc - tc) = Rd^(-1)*(Od - td)
    =>
    Since Rc and Rd are rotations, they are orthogonal matrices, so Rc^(-1) = Rc^T and Rd^(-1) = Rd^T
    =>
    Rc.T*(Oc - tc) = Rd.T*(Od - td)
    => 
    Oc = Rc*Rd.T*(Od - td) + tc
    => 
    Oc = Rc*Rd.T*Od - Rc*Rd.T*td + tc
    =>
    R = Rc*Rd.T
    t = tc - Rc*Rd.T*td
    """
    R = Rc @ Rd.T
    t = tc - R @ td
    t = t.reshape(3,1)

    # For every pixel in the depth image, find the corresponding pixel in the color image and sample the color.
    for v in range(h):
        for u in range(w):
            Z = depth[v, u] / 1000.0 # convert to meters
            if Z == 0:
                continue

            # 3D in depth camera
            X = (u - cx_d) * Z / fx_d
            Y = (v - cy_d) * Z / fy_d
            P_d = np.array([[X], [Y], [Z]])

            # Transform to color camera
            P_c = R @ P_d + t
            Xc, Yc, Zc = P_c.flatten()

            if Zc <= 0: # ?? check this
                print("x", end="", flush=True)
                continue

            # Project to color image
            u_c = int(fx_c * Xc / Zc + cx_c)
            v_c = int(fy_c * Yc / Zc + cy_c)

            # Sample color
            if 0 <= u_c < color.shape[1] and 0 <= v_c < color.shape[0]:
                color_out[v, u] = color[v_c, u_c]

    return color_out, depth



def create_pcl(rgb_frame: np.ndarray, depth_frame: np.ndarray, camera_params: dict) -> o3d.geometry.PointCloud:
    """
    Creates a point cloud from the rgb and depth frames.
    """
    # FIXME: add undistortion
    world2cam = np.array(camera_params["word2cam"])

    Kc = np.array(camera_params["K"])   
    fx_c, fy_c = Kc[0,0], Kc[1,1]
    cx_c, cy_c = Kc[0,2], Kc[1,2]

    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frame),
            o3d.geometry.Image(np.ascontiguousarray(depth_frame)),
            depth_scale=1000.0,
            depth_trunc=10,
            convert_rgb_to_intensity=False,
        ),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=rgb_frame.shape[1], height=rgb_frame.shape[0], fx=fx_c, fy=fy_c, cx=cx_c, cy=cy_c
        ),
        # extrinsic=world2cam
    )
    return pcd


def create_pcl_from_depth(depth_frame: np.ndarray, camera_params: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Creates a point cloud from the depth frame.
    """

    fx_d, fy_d = camera_params["depth_K"][0,0], camera_params["depth_K"][1,1]
    cx_d, cy_d = camera_params["depth_K"][0,2], camera_params["depth_K"][1,2]

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=o3d.geometry.Image(depth_frame),
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=depth_frame.shape[1], height=depth_frame.shape[0], fx=fx_d, fy=fy_d, cx=cx_d, cy=cy_d
        ),
        extrinsic=np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
        depth_scale=1000.0,
        depth_trunc=10
    )
    return pcd


def integrate_pointclouds_tsdf(
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    camera_params_list: list[dict],
    volume_length: float = 10.0,
    voxel_length: float = 0.006,
    sdf_trunc: float = 0.04,
    depth_scale: float = 1000.0,
    depth_trunc: float = 10.0
) -> o3d.geometry.PointCloud:
    """
    Integrates multiple point clouds into one using TSDF (Truncated Signed Distance Function).
    
    Args:
        rgb_frames: List of RGB images (BGR format from OpenCV).
        depth_frames: List of depth images (uint16, depth in millimeters).
        camera_params_list: List of camera parameter dictionaries, each containing:
            - "K": camera intrinsic matrix
            - "word2cam": world-to-camera transformation matrix (4x4)
        volume_length: Length of the TSDF volume in meters.
        voxel_length: Size of each voxel in meters.
        sdf_trunc: Truncation value for the signed distance function.
        depth_scale: Scale factor to convert depth values to meters (1000.0 for mm to m).
        depth_trunc: Maximum depth value to consider in meters.
    
    Returns:
        A single integrated point cloud extracted from the TSDF volume.
    """
    
    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Integrate each RGBD image into the TSDF volume
    for rgb_frame, depth_frame, camera_params in zip(rgb_frames, depth_frames, camera_params_list):
        assert rgb_frame.shape[:2] == depth_frame.shape[:2], \
            f"RGB and depth frames must have the same dimensions. Got RGB: {rgb_frame.shape}, Depth: {depth_frame.shape}"
        
        # Convert BGR to RGB for Open3D
        rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_frame_rgb),
            o3d.geometry.Image(np.ascontiguousarray(depth_frame)),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # Get camera intrinsic matrix
        K = np.array(camera_params["K"])
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        width, height = rgb_frame.shape[1], rgb_frame.shape[0]
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        
        # Get camera extrinsic (world-to-camera transformation)
        world2cam = np.array(camera_params["word2cam"])
        assert world2cam.shape == (4, 4), \
            f"world2cam must be a 4x4 matrix, got shape {world2cam.shape}"
        
        # Integrate into TSDF volume
        volume.integrate(rgbd, intrinsic, world2cam)
    
    # Extract point cloud from TSDF volume
    mesh = volume.extract_triangle_mesh()
    pcd = mesh.sample_points_uniformly(number_of_points=100000)
    
    assert len(pcd.points) > 0, "Integrated point cloud should contain at least one point"
    return pcd


if __name__ == "__main__":
    # rgb_frame = cv2.imread("full_depth/Aoutput0_600_rgb.png")
    # depth_frame = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)


    # K = CAMERA_PARAMS["Aoutput0"]["K"]
    # params = {"fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2]}
    # pcd = create_pcl_from_depth(depth_frame, params)
    # o3d.visualization.draw_geometries([pcd])

    # undistorted, newcameramtx = undistort(rgb_frame, np.array(CAMERA_PARAMS["Aoutput0"]["K"]), np.array(CAMERA_PARAMS["Aoutput0"]["dist"]))
    # dist_flow = viz_dist_flow(rgb_frame, np.array(CAMERA_PARAMS["Aoutput0"]["K"]), np.array(CAMERA_PARAMS["Aoutput0"]["dist"]))
    # # cv2.imwrite("samples/dist_flow.png", dist_flow)
    # cv2.imwrite("samples/rgb_frame.png", rgb_frame)
    # cv2.imwrite("samples/undistorted_rgb_frame.png", undistorted)
    # cv2.imshow("RGB Frame", rgb_frame)
    # cv2.imshow("Undistorted RGB Frame", undistorted)
    # cv2.imshow("Distortion Flow", dist_flow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Depth Viz", depth_viz)

    # color_aligned, depth_frame = align_color_to_depth(depth_frame, rgb_frame, CAMERA_PARAMS["Aoutput0"])
    # cv2.imwrite("samples/distorted_aligned_color.png", color_aligned)
    # cv2.imshow("Color Aligned", color_aligned)
    # depth_viz = viz_depth(depth_frame)
    # # cv2.imwrite("samples/distorted_aligned_depth.png", depth_viz)
    # cv2.imshow("Depth Viz", depth_viz)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pcd = create_pcl(color_aligned, depth_frame, CAMERA_PARAMS["Aoutput0"])
    # o3d.visualization.draw_geometries([pcd])


    rgb_frames = []
    depth_frames = []
    camera_params_list = []
    pcds = []

    for camera_name in ALL_CAMERAS:
        if camera_name == "hololens":
            continue
        
        print(camera_name)

        rgb = cv2.imread(f"full_depth/{camera_name}_600_rgb.png")
        depth = cv2.imread(f"full_depth/{camera_name}_600_depth.png", cv2.IMREAD_UNCHANGED)
        rgb, depth = align_color_to_depth(depth, rgb, CAMERA_PARAMS[camera_name])
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        camera_params_list.append(CAMERA_PARAMS[camera_name])
        # pcd = create_pcl(rgb, depth, CAMERA_PARAMS[camera_name])
        # o3d.visualization.draw_geometries([pcd])
        

    pcd = integrate_pointclouds_tsdf(rgb_frames, depth_frames, camera_params_list)
    o3d.visualization.draw_geometries([pcd])