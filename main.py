import cv2
import numpy as np
from classes.LaplacianBlending import LaplacianBlending
from classes.ImageTools import ImageTools
from classes.DetectionTools import DetectionTools

# Function for calling Laplacian Blend
def LaplacianBlend(l, r, m):
    lb = LaplacianBlending(l, r, m, 20)  # Adjust the last integer to adjust the number of pyramid levels
    return lb.blend()

DEBUG = False

# Scale size
scaleSize = (400, 225)

def main():

    deviceId1 = 0  # Default camera ID
    deviceId2 = 1  # Secondary camera ID

    # Open the first webcam
    cap1 = cv2.VideoCapture(deviceId1)
    if not cap1.isOpened():
        print(f"Capture Device ID {deviceId1} cannot be opened.")
        return

    # Open the second webcam
    cap2 = cv2.VideoCapture(deviceId2)
    if not cap2.isOpened():
        print(f"Capture Device ID {deviceId2} cannot be opened.")
        return

    # Initialize windows
    Window1 = "Webcam 1"
    cv2.namedWindow(Window1)

    Window2 = "Webcam 2"
    cv2.namedWindow(Window2)

    # Handle delay for camera initialization
    cv2.waitKey(1000)

    # Find the camera equalization transform
    orig, match, tMatrix = None, None, None
    IT = ImageTools()
    foundGoodTransform = False
    possibleTransform = False

    while not foundGoodTransform:
        ret1, match = cap1.read()
        ret2, orig = cap2.read()

        orig = cv2.resize(orig, scaleSize)
        match = cv2.resize(match, scaleSize)

        useKeypoints = True
        possibleTransform, tMatrix = IT.findGainTransform(orig, match, useKeypoints)

        if not possibleTransform:
            print("Trying again...")
            continue

        # convert the orig image to the dtype uf32 or uf64
        orig2 = orig.astype(np.float32)
        print(orig2.shape)

        # Test transform
        img_transform = IT.applyGainTransform(orig2, tMatrix)

        cv2.imshow("Camera 1", orig)
        cv2.imshow("Camera 2", match)
        cv2.imshow("Transformed image", img_transform)
        cv2.waitKey(200)

        userInput = input("Is this a good transform? (y/n): ")
        if userInput == "y":
            foundGoodTransform = True

    print("Starting lip flip session...")
    cv2.destroyAllWindows()

    frame1, frame2 = None, None
    d1 = DetectionTools()
    d2 = DetectionTools()

    frame1_temp, frame2_temp = None, None
    blend_range = 6
    t = 0

    while True:
        t = cv2.getTickCount()

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        frame1 = cv2.resize(frame1, scaleSize)
        frame2 = cv2.resize(frame2, scaleSize)

        # Apply the gain matching transform to the second frame
        frame2 = IT.applyGainTransform(frame2, tMatrix)

        frame2.copyTo(frame2_temp)
        frame1.copyTo(frame1_temp)

        d1.findLipRegion(frame1)
        d2.findLipRegion(frame2)

        if not d1.mouthROI.empty() and not d2.mouthROI.empty():
            cols1 = d1.mouthROI.shape[1]
            cols2 = d2.mouthROI.shape[1]

            scaledMouth1 = cv2.resize(d1.mouthROI, (cols2, d1.mouthROI.shape[0]))
            scaledMouth2 = cv2.resize(d2.mouthROI, (cols1, d2.mouthROI.shape[0]))

            if d1.mouthLoc[1] + scaledMouth2.shape[0] <= frame1.shape[0] and d1.mouthLoc[0] + scaledMouth2.shape[1] <= frame1.shape[1]:
                scaledMouth2.copyTo(frame1[d1.mouthLoc[1]:d1.mouthLoc[1] + scaledMouth2.shape[0], d1.mouthLoc[0]:d1.mouthLoc[0] + scaledMouth2.shape[1]])

                l8u_1 = frame1
                r8u_1 = frame1_temp
                l_1 = l8u_1.astype(np.float32) / 255.0
                r_1 = r8u_1.astype(np.float32) / 255.0

                m_1 = np.zeros(l_1.shape, dtype=np.float32)
                mouthCenter1 = (d1.mouthLoc[0] + 0.5 * scaledMouth2.shape[1], d1.mouthLoc[1] + 0.5 * scaledMouth2.shape[0])
                mouthAxis1 = (scaledMouth2.shape[1] * 0.5 - blend_range, scaledMouth2.shape[0] * 0.5 - blend_range)
                cv2.ellipse(m_1, (int(mouthCenter1[0]), int(mouthCenter1[1])), (int(mouthAxis1[0]), int(mouthAxis1[1])), 0, 0, 360, 1, -1)

                blend_1 = LaplacianBlend(l_1, r_1, m_1)

                if DEBUG:
                    cv2.rectangle(blend_1, (d1.rectMouth[0], d1.rectMouth[1]), (d1.rectMouth[0] + d1.rectMouth[2], d1.rectMouth[1] + d1.rectMouth[3]), (0, 255, 0), 1)
                    cv2.rectangle(blend_1, (d1.rectMouthCascade[0], d1.rectMouthCascade[1]), (d1.rectMouthCascade[0] + d1.rectMouthCascade[2], d1.rectMouthCascade[1] + d1.rectMouthCascade[3]), (0, 0, 255), 1)

                cv2.imshow(Window1, (blend_1 * 255).astype(np.uint8))

            if d2.mouthLoc[1] + scaledMouth1.shape[0] <= frame2.shape[0] and d2.mouthLoc[0] + scaledMouth1.shape[1] <= frame2.shape[1]:
                scaledMouth1.copyTo(frame2[d2.mouthLoc[1]:d2.mouthLoc[1] + scaledMouth1.shape[0], d2.mouthLoc[0]:d2.mouthLoc[0] + scaledMouth1.shape[1]])

                l8u_2 = frame2
                r8u_2 = frame2_temp
                l_2 = l8u_2.astype(np.float32) / 255.0
                r_2 = r8u_2.astype(np.float32) / 255.0

                m_2 = np.zeros(l_2.shape, dtype=np.float32)
                mouthCenter2 = (d2.mouthLoc[0] + 0.5 * scaledMouth1.shape[1], d2.mouthLoc[1] + 0.5 * scaledMouth1.shape[0])
                mouthAxis2 = (scaledMouth1.shape[1] * 0.5 - blend_range, scaledMouth1.shape[0] * 0.5 - blend_range)
                cv2.ellipse(m_2, (int(mouthCenter2[0]), int(mouthCenter2[1])), (int(mouthAxis2[0]), int(mouthAxis2[1])), 0, 0, 360, 1, -1)

                blend_2 = LaplacianBlend(l_2, r_2, m_2)

                if DEBUG:
                    cv2.rectangle(blend_2, (d2.rectMouthCascade[0], d2.rectMouthCascade[1]), (d2.rectMouthCascade[0] + d2.rectMouthCascade[2], d2.rectMouthCascade[1] + d2.rectMouthCascade[3]), (0, 0, 255), 1)
                    cv2.rectangle(blend_2, (d2.rectMouth[0], d2.rectMouth[1]), (d2.rectMouth[0] + d2.rectMouth[2], d2.rectMouth[1] + d2.rectMouth[3]), (0, 255, 0), 1)

                cv2.imshow(Window2, (blend_2 * 255).astype(np.uint8))

            t = cv2.getTickCount() - t
            print(f"{t / (cv2.getTickFrequency() * 1000.0):.2f} ms")

        else:
            cv2.imshow(Window2, frame2_temp)
            cv2.imshow(Window1, frame1_temp)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
