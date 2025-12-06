import cv2
import numpy as np


def align_color_to_depth(depth, color, Kd, dist_d, Kc, dist_c, R, t):
    # Undistort images
    depth_und = cv2.undistort(depth, Kd, dist_d)
    color_und = cv2.undistort(color, Kc, dist_c)

    h, w = depth_und.shape
    color_out = np.zeros((h, w, 3), dtype=np.uint8)

    fx_d, fy_d = Kd[0,0], Kd[1,1]
    cx_d, cy_d = Kd[0,2], Kd[1,2]

    fx_c, fy_c = Kc[0,0], Kc[1,1]
    cx_c, cy_c = Kc[0,2], Kc[1,2]

    R = np.array(R).reshape(3,3)
    t = np.array(t).reshape(3,1)

    for v in range(h):
        for u in range(w):
            Z = depth_und[v, u]
            if Z == 0:
                continue

            # 3D in depth camera
            X = (u - cx_d) * Z / fx_d
            Y = (v - cy_d) * Z / fy_d
            P_d = np.array([[X], [Y], [Z]])

            # Transform to color camera
            P_c = R @ P_d + t
            Xc, Yc, Zc = P_c[:,0]

            if Zc <= 0:
                continue

            # Project to color image
            u_c = int(fx_c * Xc / Zc + cx_c)
            v_c = int(fy_c * Yc / Zc + cy_c)

            # Sample color
            if 0 <= u_c < color_und.shape[1] and 0 <= v_c < color_und.shape[0]:
                color_out[v, u] = color_und[v_c, u_c]

    return color_out
