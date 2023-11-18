import cv2
import numpy as np
from classes.DetectionTools import DetectionTools
from classes.LaplacianBlending import LaplacianBlending
from classes.ImageTools import ImageTools

# # Use all libraries because I'm lazy.
# print("OpenCV version:", cv2.__version__)
# print("OpenCV build information:", cv2.getBuildInformation())

# Function for calling Laplacian Blend
def laplacian_blend(l, r, m):
    lb = LaplacianBlending(l, r, m, 20)  # Adjust the last integer to adjust the number of pyramid levels
    return lb.blend()

DEBUG = False

# SCALE SIZE
# scale_size = (320, 180)
# scale_size = (640, 360)
scale_size = (400, 225)

# Default camera IDs
device_id1 = 0
device_id2 = 1

cap1 = cv2.VideoCapture(device_id1)
if not cap1.isOpened():
    print(f"Capture Device ID {device_id1} cannot be opened.")
    exit()

cap2 = cv2.VideoCapture(device_id2)
if not cap2.isOpened():
    print(f"Capture Device ID {device_id2} cannot be opened.")
    exit()

window1 = "Webcam 1"
cv2.namedWindow(window1)

window2 = "Webcam 2"
cv2.namedWindow(window2)

# WTF kyle's webcam is dumb
cv2.waitKey(1000)

# Find the camera equalization transform
orig, match, t_matrix = None, None, None
it = ImageTools()

found_good_transform = False
possible_transform = False

while not found_good_transform:
    _, match = cap1.read()
    _, orig = cap2.read()

    # Scale image down
    orig = cv2.resize(orig, scale_size)
    match = cv2.resize(match, scale_size)

    use_keypoints = 1
    possible_transform = it.find_gain_transform(orig, match, t_matrix, use_keypoints)

    if not possible_transform:
        print("Trying again...")
        continue

    # Test transform
    img_transform = it.apply_gain_transform(orig, t_matrix)

    # TODO: Display images strategically on the screen, or perhaps display them side by side on the same window?

    cv2.imshow("Camera 1", orig)
    cv2.imshow("Camera 2", match)
    cv2.imshow("Transformed image", img_transform)
    cv2.waitKey(200)

    user_input = ""
    while user_input not in ["y", "n"]:
        user_input = input("Is this a good transform? (y/n): ").lower()
        if user_input == "y":
            found_good_transform = True

print("Starting lip flip session...")

# TODO: save transform as a text file and have the option to load it up instead of doing this calibration every session.

cv2.destroyAllWindows()

frame1 = None  # Holds the current frame from the Video device:
frame2 = None

d1 = DetectionTools()
d2 = DetectionTools()

frame1_temp = None
frame2_temp = None

blend_range = 6
t = 0

while True:
    t = cv2.getTickCount()

    _, frame1 = cap1.read()
    _, frame2 = cap2.read()

    frame1 = cv2.resize(frame1, scale_size)
    frame2 = cv2.resize(frame2, scale_size)

    # Apply the gain matching transform
    frame2 = it.apply_gain_transform(frame2, t_matrix)

    frame2_temp = frame2.copy()
    frame1_temp = frame1.copy()

    d1.find_lip_region(frame1)
    d2.find_lip_region(frame2)

    if not d1.mouth_roi.empty() and not d2.mouth_roi.empty():
        # Scale mouths to match the corresponding mouth (use columns so that the "openMouth" capability works)
        cols1 = float(d1.mouth_roi.shape[1])
        cols2 = float(d2.mouth_roi.shape[1])

        scaled_mouth1 = cv2.resize(d1.mouth_roi, (int(cols2 / cols1 * d1.mouth_roi.shape[1]), int(cols2 / cols1 * d1.mouth_roi.shape[0])))
        scaled_mouth2 = cv2.resize(d2.mouth_roi, (int(cols1 / cols2 * d2.mouth_roi.shape[1]), int(cols1 / cols2 * d2.mouth_roi.shape[0])))

        if d1.mouth_loc[1] + scaled_mouth2.shape[0] <= frame1.shape[0] and d1.mouth_loc[0] + scaled_mouth2.shape[1] <= frame1.shape[1]:
            # Copy mouth over
            frame1[d1.mouth_loc[1]:d1.mouth_loc[1] + scaled_mouth2.shape[0],
            d1.mouth_loc[0]:d1.mouth_loc[0] + scaled_mouth2.shape[1]] = scaled_mouth2

            # Blending
            l8u_1 = frame1
            r8u_1 = frame1_temp
            l_1 = l8u_1.astype(np.float32) / 255.0
            r_1 = r8u_1.astype(np.float32) / 255.0

            m_1 = np.zeros_like(l_1, dtype=np.float32)
            mouth_center1 = (d1.mouth_loc[0] + 0.5 * scaled_mouth2.shape[1], d1.mouth_loc[1] + 0.5 * scaled_mouth2.shape[0])
            mouth_axis1 = (scaled_mouth2.shape[1] * 0.5 - blend_range, scaled_mouth2.shape[0] * 0.5 - blend_range)
            cv2.ellipse(m_1, (int(mouth_center1[0]), int(mouth_center1[1])), (int(mouth_axis1[0]), int(mouth_axis1[1])),
                        0, 0, 360, 1, -1)

            blend_1 = laplacian_blend(l_1, r_1, m_1)

            if DEBUG:
                cv2.rectangle(blend_1, tuple(d1.rect_mouth[0]), tuple(d1.rect_mouth[1]), (0, 255, 0), 1)
                cv2.rectangle(blend_1, tuple(d1.rect_mouth_cascade[0]), tuple(d1.rect_mouth_cascade[1]), (0, 0, 255), 1)

            cv2.imshow(window1, blend_1)
            if DEBUG:
                cv2.imshow("Mask1", m_1)

        if d2.mouth_loc[1] + scaled_mouth1.shape[0] <= frame2.shape[0] and d2.mouth_loc[0] + scaled_mouth1.shape[1] <= frame2.shape[1]:
            # Copy mouth over
            frame2[d2.mouth_loc[1]:d2.mouth_loc[1] + scaled_mouth1.shape[0],
            d2.mouth_loc[0]:d2.mouth_loc[0] + scaled_mouth1.shape[1]] = scaled_mouth1

            # Blending
            l8u_2 = frame2
            r8u_2 = frame2_temp
            l_2 = l8u_2.astype(np.float32) / 255.0
            r_2 = r8u_2.astype(np.float32) / 255.0

            m_2 = np.zeros_like(l_2, dtype=np.float32)
            mouth_center2 = (d2.mouth_loc[0] + 0.5 * scaled_mouth1.shape[1], d2.mouth_loc[1] + 0.5 * scaled_mouth1.shape[0])
            mouth_axis2 = (scaled_mouth1.shape[1] * 0.5 - blend_range, scaled_mouth1.shape[0] * 0.5 - blend_range)
            cv2.ellipse(m_2, (int(mouth_center2[0]), int(mouth_center2[1])), (int(mouth_axis2[0]), int(mouth_axis2[1])),
                        0, 0, 360, 1, -1)

            blend_2 = laplacian_blend(l_2, r_2, m_2)

            if DEBUG:
                cv2.rectangle(blend_2, tuple(d2.rect_mouth_cascade[0]), tuple(d2.rect_mouth_cascade[1]), (0, 0, 255), 1)
                cv2.rectangle(blend_2, tuple(d2.rect_mouth[0]), tuple(d2.rect_mouth[1]), (0, 255, 0), 1)

            cv2.imshow(window2, blend_2)
            if DEBUG:
                cv2.imshow("Mask2", m_2)

        t = cv2.getTickCount() - t
        print(f"{t / (cv2.getTickFrequency() * 1000.0)} ms")

    else:
        cv2.imshow(window2, frame2_temp)
        cv2.imshow(window1, frame1_temp)

    # Display the frames
    key = cv2.waitKey(20)
    # Exit the loop on the 'Esc' key
    if key == 27:
        break

# Release the capture devices
cap1.release()
cap2.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
