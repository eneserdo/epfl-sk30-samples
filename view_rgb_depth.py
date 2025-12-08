raise "this code is depreciated, might be removed"

import argparse
import datetime
import os
import pathlib
import sys

import cv2
import numpy as np
import open3d as o3d


def getBlended(img1, img2, multiply):
    # im1 = np.clip(img1+50, 0, 255)

    im1 = (np.sum(img1, 2) // 3).astype(np.uint8)
    im1 = cv2.merge([np.zeros_like(im1), np.zeros_like(im1), im1])

    # clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # im2=clahe.apply(img2)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    old_min = img2.min()
    old_max = img2.max()
    im2 = img2.astype(np.float32) * multiply

    im2 = np.clip(im2, old_min, old_max)

    im2 = cv2.normalize(
        im2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    im2 = im2.astype(np.uint8)

    im2 = cv2.merge([im2, np.zeros_like(im2), np.zeros_like(im2)])

    return im2 + im1  # since they are in different channels, no need for overflow check


def shift(img, ox, oy):
    def non(s):
        return s if s < 0 else None

    def mom(s):
        return max(0, s)

    shift_img = np.zeros_like(img)
    shift_img[mom(oy) : non(oy), mom(ox) : non(ox)] = img[
        mom(-oy) : non(-oy), mom(-ox) : non(-ox)
    ]
    return shift_img


def filter_depth(depth):
    # filter out invalid depth values

    # remove nans
    depth = np.where(np.isnan(depth), 0, depth)

    # remove posinfs
    max_val = depth[depth != np.inf].max()
    depth = np.where(depth == np.inf, max_val, depth)

    # remove neginfs
    min_val = depth[depth != -np.inf].min()
    depth = np.where(depth == -np.inf, min_val, depth)

    # exclude alpha channel
    if len(depth.shape) != 2:
        if depth.shape[2] == 1:
            pass
        elif depth.shape[2] == 2:
            depth = depth[:, :, :-1]
        else:
            raise ValueError(f"Depth image has invalid shape: {depth.shape}")

    return depth


def scale_depth(depth_img_):
    depth_img = depth_img_.copy()

    mean = depth_img.mean()
    std = depth_img.std()
    max_val = depth_img.max()
    min_val = depth_img.min()

    min_val = max(mean - 2 * std, min_val)
    max_val = min(mean + 2 * std, max_val)
    depth_img = np.clip(depth_img, min_val, max_val)
    depth_img = (depth_img - min_val) / (max_val - min_val) * 255
    depth_img = depth_img.astype(np.uint8)

    return depth_img


img_depth = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)


def get_mouse_pos(event, px, py, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"Mouse position: ({px}, {py})")
        depth_value = img_depth[py, px]
        print(f"Depth value at ({px}, {py}): {depth_value}")



shift_y = 0
shift_x = 0

multiply = 1

def main():
    global shift_x, shift_y, multiply


    depth_min = np.min(img_depth)
    depth_max = np.max(img_depth)

    while True:
        # DEPTH
        depth_map = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)
        depth_map = np.clip(depth_map, depth_min, depth_max).astype(np.float32)

        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_HOT)

        depth_map = cv2.putText(depth_map, f"Depth Min: {depth_min}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        depth_map = cv2.putText(depth_map, f"Depth Max: {depth_max}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Depth", depth_map)
        cv2.setMouseCallback("Depth", get_mouse_pos)


        # RGB
        img_rgb = cv2.imread("full_depth/Aoutput0_600_rgb.png")
        cv2.imshow("RGB", img_rgb)
        

        # COMBINED
        depth_map_raw = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)
        shifted_rgb = shift(img_rgb, shift_x, shift_y)

        blended = getBlended(shifted_rgb, depth_map_raw, multiply)

        blended = cv2.putText(
            blended,
            f"shift x:{2 * shift_x}, y: {2 * shift_y}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("RGB-D", blended)




        key = cv2.waitKey(5) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("d"):
            if shift_x < 200:
                shift_x = shift_x + 1

        elif key == ord("a"):
            if shift_x > -200:
                shift_x = shift_x - 1

        elif key == ord("s"):
            if shift_y < 200:
                shift_y = shift_y + 1

        elif key == ord("w"):
            if shift_y > -200:
                shift_y = shift_y - 1

        elif key == ord("r"):
            shift_x = 0
            shift_y = 0
            
        elif key == ord("z"):
            depth_min = depth_min - 5
        elif key == ord("x"):
            depth_min = depth_min + 5

        elif key == ord("c"):
            depth_max = depth_max - 5
        elif key == ord("v"):
            depth_max = depth_max + 5
        
        elif key == ord("m"):
            multiply = min(multiply + 0.1, 10)
        elif key == ord("n"):
            multiply = max(multiply - 0.1, 0)


if __name__ == "__main__":
    main()