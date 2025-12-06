# as I move my cursor over the depth map, I want to see the depth value at that point

import cv2
import numpy as np

depth_map = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)

depth_max = np.max(depth_map)
depth_min = np.min(depth_map)


def get_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse position: ({x}, {y})")
        depth_value = depth_map[y, x]
        print(f"Depth value at ({x}, {y}): {depth_value}")

while True:
    depth_map = cv2.imread("full_depth/Aoutput0_600_depth.png", cv2.IMREAD_UNCHANGED)
    depth_map = np.clip(depth_map, depth_min, depth_max).astype(np.float32)

    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_HOT)

    depth_map = cv2.putText(depth_map, f"Depth Min: {depth_min}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    depth_map = cv2.putText(depth_map, f"Depth Max: {depth_max}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Depth Map", depth_map)
    cv2.setMouseCallback("Depth Map", get_mouse_position)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    if key == ord("z"):
        depth_min = depth_min - 5
    if key == ord("x"):
        depth_min = depth_min + 5

    if key == ord("a"):
        depth_max = depth_max - 5
    if key == ord("s"):
        depth_max = depth_max + 5

cv2.destroyAllWindows()