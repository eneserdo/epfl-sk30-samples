import glob
import os
from typing import Any

import cv2
import numpy as np

video_path = "/home/enerd/epfl-sk30-samples/2023_08_11_14_15_21/videos"

video_files = glob.glob(os.path.join(video_path, "*.mp4"))

frame_idx = 1000
all_frames = {}

for video_file in video_files:
    fname = os.path.basename(video_file)
    video = cv2.VideoCapture(video_file)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video.read()
    if not ret:
        print(f"Failed to read frame {frame_idx} from {video_file}")
        continue

    all_frames[fname] = frame

# combine all frames into a single image 5 by 2 grid
h, w, _ = list(all_frames.values())[0].shape
gap = 5

combined_image = np.zeros((5 * h+4*gap, 2 * w+gap, 3), dtype=np.uint8)

for i, (fname, frame) in enumerate(all_frames.items()):
    print(fname)
    if "holo" in fname:
        frame = cv2.resize(frame, (w, h))

    row, col = divmod(i, 2)
    top = row * (h+gap)
    bottom = top + h
    left = col * w
    right = left + w
    combined_image[top:bottom, left:right] = frame

cv2.imwrite("combined_image.jpg", combined_image)